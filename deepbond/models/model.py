import logging
from abc import ABCMeta, abstractmethod

import torch

logger = logging.getLogger(__name__)


class Model(torch.nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self, words_field, tags_field):
        super().__init__()
        # Default fields and embeddings
        self.words_field = words_field
        self.tags_field = tags_field
        # Building flag
        self.is_built = False
        # Loss function has to be defined in build()
        self._loss = None

    @property
    def nb_classes(self):
        return len(self.tags_field.vocab.stoi) - 1  # remove pad index

    def loss(self, pred, gold):
        # (bs*ts, nb_classes)
        predicted = pred.reshape(-1, self.nb_classes)

        # (bs*ts, )
        gold = gold.reshape(-1)

        return self._loss(predicted, gold)

    @abstractmethod
    def build(self, **params):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def predict_proba(self, batch):
        pred = self.forward(batch)
        return torch.exp(pred)  # assume log softmax in the output

    def predict_classes(self, batch):
        classes = torch.argmax(self.predict_proba(batch), -1)
        return classes

    def load(self, path):
        logger.info("Loading model weights from {}".format(path))
        self.load_state_dict(
            torch.load(str(path), map_location=lambda storage, loc: storage)
        )

    def save(self, path):
        logger.info("Saving model weights to {}".format(path))
        torch.save(self.state_dict(), str(path))
