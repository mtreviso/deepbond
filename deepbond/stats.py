import numpy as np
import torch

from deepbond import constants
from deepbond.models.utils import unroll, unmask

from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef


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
    def __init__(self, tags_vocab):
        self.mask_id = constants.TAGS_PAD_ID
        self.tags_vocab = tags_vocab

        # this attrs will be updated every time a new prediction is added
        self.pred_classes = []
        self.pred_probs = []
        self.golds = []
        self.loss = 0

        # this attrs will be set when get_ methods are called
        self.prec_rec_f1 = None
        self.ser = None
        self.mcc = None

        # this attrs will be set when calc method is called
        self.best_prec_rec_f1 = BestValueEpoch(value=[0, 0, 0], epoch=1)
        self.best_ser = BestValueEpoch(value=float('inf'), epoch=1)
        self.best_mcc = BestValueEpoch(value=0, epoch=1)
        self.best_loss = BestValueEpoch(value=float('inf'), epoch=1)

        # private (used for lazy calculation)
        self._flattened_preds = None
        self._flattened_golds = None

    def reset(self):
        self.pred_classes.clear()
        self.pred_probs.clear()
        self.golds.clear()
        self.loss = 0
        self.prec_rec_f1 = None
        self.ser = None
        self.mcc = None
        self._flattened_preds = None
        self._flattened_golds = None

    @property
    def nb_batches(self):
        return len(self.golds)

    def update(self, loss, preds, golds):
        pred_probs = torch.exp(preds)  # assuming log softmax at the nn output
        pred_classes = pred_probs.argmax(dim=-1)
        self.loss += loss

        # unmask & flatten predictions and gold labels before storing them
        mask = golds != self.mask_id
        self.pred_probs.append(unroll(unmask(pred_probs, mask)))
        self.pred_classes.append(unroll(unmask(pred_classes, mask)))
        self.golds.append(unroll(unmask(golds, mask)))

    def get_loss(self):
        return self.loss / self.nb_batches

    def _get_flattened_golds_and_preds(self):
        if self._flattened_preds is None:
            self._flattened_preds = np.array(unroll(self.pred_classes))
        if self._flattened_golds is None:
            self._flattened_golds = np.array(unroll(self.golds))
        return self._flattened_golds, self._flattened_preds

    def get_prec_rec_f1(self):
        if self.prec_rec_f1 is None:
            gold_labels, pred_labels = self._get_flattened_golds_and_preds()
            prec, rec, f1, _ = precision_recall_fscore_support(
                gold_labels,
                pred_labels,
                beta=1.0,
                pos_label=self.tags_vocab['.'],
                average='binary'
            )
            self.prec_rec_f1 = [prec, rec, f1]
        return self.prec_rec_f1

    def get_slot_error_rate(self):
        if self.ser is None:
            gold_labels, pred_labels = self._get_flattened_golds_and_preds()
            slots = np.sum(gold_labels)
            errors = np.sum(gold_labels != pred_labels)
            self.ser = errors / slots
        return self.ser

    def get_mcc(self):
        if self.mcc is None:
            gold_labels, pred_labels = self._get_flattened_golds_and_preds()
            mcc = matthews_corrcoef(
                gold_labels,
                pred_labels
            )
            self.mcc = mcc
        return self.mcc

    def calc(self, current_epoch):
        current_loss = self.get_loss()
        current_prec_rec_f1 = self.get_prec_rec_f1()
        current_ser = self.get_slot_error_rate()
        current_mcc = self.get_mcc()

        if current_loss < self.best_loss.value:
            self.best_loss.value = current_loss
            self.best_loss.epoch = current_epoch

        if current_prec_rec_f1[2] > self.best_prec_rec_f1.value[2]:
            self.best_prec_rec_f1.value[0] = current_prec_rec_f1[0]
            self.best_prec_rec_f1.value[1] = current_prec_rec_f1[1]
            self.best_prec_rec_f1.value[2] = current_prec_rec_f1[2]
            self.best_prec_rec_f1.epoch = current_epoch

        if current_ser < self.best_ser.value:
            self.best_ser.value = current_ser
            self.best_ser.epoch = current_epoch

        if current_mcc > self.best_mcc.value:
            self.best_mcc.value = current_mcc
            self.best_mcc.epoch = current_epoch

    def to_dict(self):
        return {
            'loss': self.get_loss(),
            'prec_rec_f1': self.get_prec_rec_f1(),
            'ser': self.get_slot_error_rate(),
            'mcc': self.get_mcc(),
            'best_loss': self.best_loss,
            'best_prec_rec_f1': self.best_prec_rec_f1,
            'best_ser': self.best_ser,
            'best_mcc': self.best_mcc,
        }
