from pathlib import Path
from typing import *
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from functools import partial
from overrides import overrides
import pickle
import json

from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.nn import util as nn_util

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import DataIterator
from tqdm import tqdm
from scipy.special import expit # the sigmoid function
import allennlp
from allennlp.data.fields import TextField, MetadataField, ArrayField
from allennlp.data.iterators import BasicIterator
from allennlp.common.checks import ConfigurationError

from allennlp.data.token_indexers import PretrainedBertIndexer

import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.training.trainer import Trainer

from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder

from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.training.metrics import CategoricalAccuracy


test_df = pd.read_csv('../emotion_data/emotions_labelled_data_test.csv')
label_cols = list(test_df['label'].unique())
label_cols.sort()


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

config = Config(
    testing=False,
    seed=1,
    batch_size=32,
    lr=0.001,
    epochs=40,
    hidden_sz=300,
    max_seq_len=128, # necessary to limit memory usage
    max_vocab_size=100000,
)

USE_GPU = torch.cuda.is_available()
DATA_ROOT = Path("")
torch.manual_seed(config.seed)


# Load Data
class EmotionDatasetReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]]=lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int]=config.max_seq_len) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_seq_len = max_seq_len

    @overrides
    def text_to_instance(self, tokens: List[Token], id: str=None,
                         labels: np.ndarray=None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        id_field = MetadataField(id)
        fields["id"] = id_field

        if labels is None:
            labels = np.zeros(len(label_cols))
        label_field = ArrayField(array=labels)
        fields["label"] = label_field

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        df = pd.read_csv(file_path)
        if config.testing: df = df.head(2000)
        for i, row in df.iterrows():
            yield self.text_to_instance(
                [Token(x) for x in self.tokenizer(str(row["text"]))],
                i, row[label_cols].values,
            )

token_indexer = PretrainedBertIndexer(
    pretrained_model="bert-base-uncased",
    max_pieces=config.max_seq_len,
    do_lowercase=True,
 )


def tokenizer(s: str):
    return token_indexer.wordpiece_tokenizer(s)[:config.max_seq_len - 2]


reader = EmotionDatasetReader(
    tokenizer=tokenizer,
    token_indexers={"tokens": token_indexer}
)

train_ds, test_ds = (reader.read(DATA_ROOT / fname) for fname in ["train.csv", "test.csv"])
val_ds = None

vocab = Vocabulary()

from allennlp.data.iterators import BucketIterator


iterator = BucketIterator(batch_size=config.batch_size,
                          sorting_keys=[("tokens", "num_tokens")],
                         )

iterator.index_with(vocab)

batch = next(iter(iterator(train_ds)))

# Prepare Model

class BaselineModel(Model):
    def __init__(self, word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 out_sz: int=len(label_cols)):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.projection = nn.Linear(self.encoder.get_output_dim(), out_sz)
        self.loss = nn.BCEWithLogitsLoss()
        self.accuracy = CategoricalAccuracy()

    def forward(self, tokens: Dict[str, torch.Tensor],
                id: Any, label: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        state = self.encoder(embeddings, mask)
        class_logits = self.projection(state)

        output = {"class_logits": class_logits}
        output["loss"] = self.loss(class_logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

bert_embedder = PretrainedBertEmbedder(
        pretrained_model="bert-base-uncased",
        top_layer_only=True, # conserve memory
)
word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                                            # we'll be ignoring masks so we'll need to set this to True
                                                           allow_unmatched_keys = True)

BERT_DIM = word_embeddings.get_output_dim()

class BertSentencePooler(Seq2VecEncoder):
    def forward(self, embs: torch.tensor,
                mask: torch.tensor=None) -> torch.tensor:
        # extract first token tensor
        return embs[:, 0]

    @overrides
    def get_output_dim(self) -> int:
        return BERT_DIM

encoder = BertSentencePooler(vocab)

model = BaselineModel(
    word_embeddings,
    encoder,
)

if USE_GPU: model.cuda()
else: model
optimizer = optim.Adam(model.parameters(), lr=config.lr)

# Train

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    iterator=iterator,
    train_dataset=train_ds,
    cuda_device=0 if USE_GPU else -1,
    num_epochs=config.epochs,
)

metrics = trainer.train()


# Predict
def tonp(tsr): return tsr.detach().cpu().numpy()


class Predictor:
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int=-1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device

    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch)
        return expit(tonp(out_dict["class_logits"]))

    def predict(self, ds: Iterable[Instance]) -> np.ndarray:
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        preds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                preds.append(self._extract_data(batch))
        return np.concatenate(preds, axis=0)

# Save Model & Results
with open("BERT_model.th", 'wb') as f:
    torch.save(model.state_dict(), f)

vocab.save_to_files("BERT_vocabulary")

with open('BERT_metrics.json', "w") as file:
    json.dump(metrics, file, indent=4)

with open('test_predictions_list.pkl', 'wb') as f:
    pickle.dump(test_prediction_list, f)

