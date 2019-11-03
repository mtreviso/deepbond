import numpy as np
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef

from deepbond import constants
from deepbond.models.utils import unroll, unmask


class BestValueEpoch:
    def __init__(self, value, epoch):
        self.value = value
        self.epoch = epoch


class Stats(object):
    """
    Keep stats information during training and evaluation

    Args:
        tags_vocab (dict): vocab object for tags field
    """
    def __init__(self, tags_vocab):
        self.tags_vocab = tags_vocab

        # this attrs will be updated every time a new prediction is added
        self.pred_classes = []
        self.gold_classes = []
        self.loss_accum = 0

        # this attrs will be set when get_ methods are called
        self.loss = None
        self.prec_rec_f1 = None
        self.ser = None
        self.mcc = None

        # this attrs will be set when calc method is called
        self.best_prec_rec_f1 = BestValueEpoch(value=[0, 0, 0], epoch=1)
        self.best_ser = BestValueEpoch(value=float('inf'), epoch=1)
        self.best_mcc = BestValueEpoch(value=0, epoch=1)
        self.best_loss = BestValueEpoch(value=float('inf'), epoch=1)

    def reset(self):
        self.pred_classes.clear()
        self.gold_classes.clear()
        self.loss_accum = 0
        self.loss = None
        self.prec_rec_f1 = None
        self.ser = None
        self.mcc = None

    @property
    def nb_batches(self):
        return len(self.gold_classes)

    def update(self, loss, pred_classes, golds):
        self.loss_accum += loss
        # unmask & flatten predictions and gold labels before storing them
        mask = golds != constants.TAGS_PAD_ID
        self.pred_classes.extend(unroll(unmask(pred_classes, mask)))
        self.gold_classes.extend(unroll(unmask(golds, mask)))

    def get_loss(self):
        return self.loss_accum / self.nb_batches

    def get_prec_rec_f1(self):
        prec, rec, f1, _ = precision_recall_fscore_support(
            self.gold_classes,
            self.pred_classes,
            beta=1.0,
            pos_label=self.tags_vocab['.'],
            average='binary'
        )
        return prec, rec, f1

    def get_slot_error_rate(self):
        slots = np.sum(self.gold_classes)
        errors = np.sum(np.not_equal(self.gold_classes, self.pred_classes))
        return errors / slots

    def get_mcc(self):
        mcc = matthews_corrcoef(self.gold_classes, self.pred_classes)
        return mcc

    def calc(self, current_epoch):
        self.loss = self.get_loss()
        self.prec_rec_f1 = self.get_prec_rec_f1()
        self.ser = self.get_slot_error_rate()
        self.mcc = self.get_mcc()

        if self.loss < self.best_loss.value:
            self.best_loss.value = self.loss
            self.best_loss.epoch = current_epoch

        if self.prec_rec_f1[2] > self.best_prec_rec_f1.value[2]:
            self.best_prec_rec_f1.value[0] = self.prec_rec_f1[0]
            self.best_prec_rec_f1.value[1] = self.prec_rec_f1[1]
            self.best_prec_rec_f1.value[2] = self.prec_rec_f1[2]
            self.best_prec_rec_f1.epoch = current_epoch

        if self.ser < self.best_ser.value:
            self.best_ser.value = self.ser
            self.best_ser.epoch = current_epoch

        if self.mcc > self.best_mcc.value:
            self.best_mcc.value = self.mcc
            self.best_mcc.epoch = current_epoch

    def to_dict(self):
        return {
            'loss': self.loss,
            'prec_rec_f1': self.prec_rec_f1,
            'ser': self.ser,
            'mcc': self.mcc,
            'best_loss': self.best_loss,
            'best_prec_rec_f1': self.best_prec_rec_f1,
            'best_ser': self.best_ser,
            'best_mcc': self.best_mcc,
        }
