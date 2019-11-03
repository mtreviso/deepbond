import torch

from deepbond import constants
from deepbond.models.utils import unmask


class Predicter:

    def __init__(self, dataset_iter, model):
        self.dataset_iter = dataset_iter
        self.model = model
        words_field = self.dataset_iter.dataset.fields['words']
        has_bos = words_field.init_token is not None
        has_eos = words_field.eos_token is not None
        self.cut_length = int(has_bos) + int(has_eos)

    def predict(self, pred_type='classes'):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in self.dataset_iter:
                mask = batch.words != constants.PAD_ID
                if pred_type == 'classes':
                    preds = unmask(self.model.predict_classes(batch),
                                   mask, cut_length=self.cut_length)
                else:
                    preds = unmask(self.model.predict_proba(batch),
                                   mask, cut_length=self.cut_length)
                predictions.extend(preds)
        return predictions
