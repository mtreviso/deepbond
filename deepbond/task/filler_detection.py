import os, re
from deepbond.loader import build_dataset_from_data
from deepbond.models.utils import get_new_texts, convert_predictions_to_tuples
from deepbond.task import Task


class FillerDetector(Task):

	def __init__(self, **kwargs):
		super(FillerDetector, self).__init__(**kwargs)
		self.options['task'] = 'dd_fillers'

