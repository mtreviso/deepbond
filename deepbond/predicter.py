import torch

from deeptagger import constants
from deeptagger.models.utils import unmask


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
                    pred = unmask(self.model.predict_classes(batch), mask)
                else:
                    pred = unmask(self.model.predict_proba(batch), mask)
                predictions.extend(pred)
        return predictions
