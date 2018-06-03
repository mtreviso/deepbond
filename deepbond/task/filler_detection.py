import os, re
from deepbond.loader import build_dataset_from_data
from deepbond.models.utils import get_new_texts, convert_predictions_to_tuples
from deepbond.task import Task


class FillerDetector(Task):

	def __init__(self, **kwargs):
		super(FillerDetector, self).__init__(**kwargs)
		self.options['task'] = 'dd_fillers'


	def restrict_wordset(self, wordset={}):
		if wordset != {}:
			self.wordset = wordset
		else:
			self.wordset = {'tá', 'ta', 'é', 'e', 'viu', 'enfim', 'sabe', 'não', 
							'olha', 'ne', 'bom', 'na', 'né', 'muito', 'então', 
							'agora', 'assim', 'tal', 'bem', 'verdade', 'tudo'}
