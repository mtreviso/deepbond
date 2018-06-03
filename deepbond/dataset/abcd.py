from deepbond.dataset import DataSet
from deepbond.helpers import Cleaner
import re

class ABCDDataSet(DataSet):
	def _clean_text_file(self, text):
		text = Cleaner.remove_punctuation(text, less='')
		text = Cleaner.lowercase(text)
		text = Cleaner.remove_newlines(text)
		text = Cleaner.trim(text)
		return text.strip()

