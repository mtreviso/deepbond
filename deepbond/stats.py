import numpy as np
import torch

from deeptagger import constants
from deeptagger.models.utils import unroll, unmask


class BestValueEpoch:
    def __init__(self, value, epoch):
        self.value = value
        self.epoch = epoch


class Stats(object):
    """
    Keep stats information during training and evaluation

    Args:
        train_vocab (dict): dict with words found in training data
        emb_vocab (dict): dict with words found in embeddings data
    """
    def __init__(self, train_vocab=None, emb_vocab=None):
        if train_vocab is None:
            train_vocab = dict()
        if emb_vocab is None:
            emb_vocab = dict()
        self.train_vocab = train_vocab
        self.emb_vocab = emb_vocab
        self.mask_id = constants.TAGS_PAD_ID

        # this attrs will be updated every time a new prediciton is added
        self.pred_classes = []
        self.pred_probs = []
        self.golds = []
        self.loss = 0

        # this attrs will be set when get_ methods are called
        self.acc = None
        self.acc_oov = None
        self.acc_emb = None
        self.acc_sent = None

        # this attrs will be se when calc method is called
        self.best_acc = BestValueEpoch(value=0, epoch=0)
        self.best_acc_oov = BestValueEpoch(value=0, epoch=0)
        self.best_acc_emb = BestValueEpoch(value=0, epoch=0)
        self.best_acc_sent = BestValueEpoch(value=0, epoch=0)
        self.best_loss = BestValueEpoch(value=np.Inf, epoch=0)

        # private (used for lazy calculation)
        self._flattened_preds = None
        self._flattened_golds = None
        self._pred_classes_sent = []
        self._golds_sent = []

    def reset(self):
        self.pred_classes.clear()
        self.pred_probs.clear()
        self._pred_classes_sent.clear()
        self._golds_sent.clear()
        self.golds.clear()
        self.loss = 0
        self.acc = None
        self.acc_oov = None
        self.acc_emb = None
        self.acc_sent = None
        self._flattened_preds = None
        self._flattened_golds = None

    @property
    def nb_batches(self):
        return len(self.golds)

    def update(self, loss, preds, golds):
        mask = golds != self.mask_id
        pred_probs = torch.exp(preds)
        pred_classes = pred_probs.argmax(dim=-1)
        self.loss += loss
        self.pred_probs.append(unroll(unmask(pred_probs, mask)))
        self._pred_classes_sent.append(unmask(pred_classes, mask))
        self.pred_classes.append(unroll(self._pred_classes_sent[-1]))
        self._golds_sent.append(unmask(golds, mask))
        self.golds.append(unroll(self._golds_sent[-1]))

    def get_loss(self):
        return self.loss / self.nb_batches

    def _get_bins(self):
        if self._flattened_preds is None:
            self._flattened_preds = np.array(unroll(self.pred_classes))
        if self._flattened_golds is None:
            self._flattened_golds = np.array(unroll(self.golds))
        return self._flattened_preds == self._flattened_golds

    def get_acc(self):
        if self.acc is None:
            bins = self._get_bins()
            self.acc = bins.mean()
        return self.acc

    def get_acc_oov(self, words=None):
        if self.acc_oov is None:
            idx = [i for i, w in enumerate(unroll(words))
                   if w not in self.train_vocab or w == constants.UNK]
            if len(idx) == 0:
                self.acc_oov = 1.0
                return self.acc_oov
            bins = self._get_bins()
            self.acc_oov = bins[idx].mean()
        return self.acc_oov

    def get_acc_emb(self, words=None):
        if self.acc_emb is None:
            idx = [i for i, w in enumerate(unroll(words))
                   if w not in self.emb_vocab and w != constants.UNK]
            if len(idx) == 0:
                self.acc_emb = 1.0
                return self.acc_emb
            bins = self._get_bins()
            self.acc_emb = bins[idx].mean()
        return self.acc_emb

    def get_acc_sentence(self):
        if self.acc_sent is None:
            bins = []
            for bx, by in zip(self._pred_classes_sent, self._golds_sent):
                for sentx, senty in zip(bx, by):
                    bins.append(sentx == senty)
            self.acc_sent = np.mean(bins)
        return self.acc_sent

    def calc(self, current_epoch, words):
        specials = [constants.PAD, constants.START, constants.STOP]
        words = list(filter(lambda w: w not in specials, unroll(words)))
        current_loss = self.get_loss()
        current_acc = self.get_acc()
        current_acc_oov = self.get_acc_oov(words)
        current_acc_emb = self.get_acc_emb(words)
        current_acc_sent = self.get_acc_sentence()

        if current_loss < self.best_loss.value:
            self.best_loss.value = current_loss
            self.best_loss.epoch = current_epoch

        if current_acc > self.best_acc.value:
            self.best_acc.value = current_acc
            self.best_acc.epoch = current_epoch

        if current_acc_oov > self.best_acc_oov.value:
            self.best_acc_oov.value = current_acc_oov
            self.best_acc_oov.epoch = current_epoch

        if current_acc_emb > self.best_acc_emb.value:
            self.best_acc_emb.value = current_acc_emb
            self.best_acc_emb.epoch = current_epoch

        if current_acc_sent > self.best_acc_sent.value:
            self.best_acc_sent.value = current_acc_sent
            self.best_acc_sent.epoch = current_epoch

    def to_dict(self):
        return {
            'loss': self.get_loss(),
            'acc': self.get_acc(),
            'acc_oov': self.get_acc_oov(),
            'acc_emb': self.get_acc_emb(),
            'acc_sent': self.get_acc_sentence(),
            'best_loss': self.best_loss,
            'best_acc': self.best_acc,
            'best_acc_oov': self.best_acc_oov,
            'best_acc_emb': self.best_acc_emb,
            'best_acc_sent': self.best_acc_sent
        }
