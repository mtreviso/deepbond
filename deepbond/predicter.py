import torch

from deepbond import constants
from deepbond.models.utils import unmask, remove_bos_and_eos_symbols


class Predicter:

    def __init__(self, dataset_iter, model):
        self.dataset_iter = dataset_iter
        self.model = model

    def predict(self, pred_type='classes'):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in self.dataset_iter:
                mask = batch.words != constants.PAD_ID
                if pred_type == 'classes':
                    preds = unmask(self.model.predict_classes(batch), mask)
                else:
                    preds = unmask(self.model.predict_proba(batch), mask)
                preds = remove_bos_and_eos_symbols(preds)
                predictions.extend(preds)
        return predictions
