from deepbond.dataset import DataSet
from deepbond.helpers import Cleaner
import re

def fix(text):
	text = re.sub(r"(?u)(^|\W)[‘’′`']", r'\1"', text)
	text = re.sub(r"(?u)[‘’`′'](\W|$)", r'"\1', text)
	text = re.sub(r'(?u)[‘’`′“”]', '"', text)
	text = re.sub(r'(?<!\.)\.\.(?!\.)', '.', text)
	text = re.sub(r' -(?=[^\W\d_])', ' - ', text)
	text = re.sub(r'\.\.\.', u'…', text)
	text = re.sub(r'(\S)([\?\!\.\,\"])', r'\1 \2', text)
	text = re.sub(r'([\?\!\.\,\"])(\S)', r'\1 \2', text)
	text = text.replace('&nbsp;', ' ')
	text = text.replace(u'“', '\"')
	text = text.replace(u'”', '\"')
	text = text.replace(u'–', '-')
	text = text.replace(u'—', '-')
	return text


class FolinhaDataSet(DataSet):
	def _clean_text_file(self, text):
		text = Cleaner.lowercase(text)
		text = Cleaner.remove_newlines(text)
		text = Cleaner.remove_tags(['author', 'date', 'link', 'subtitle', 'title', 'url'], text)
		text = fix(text)
		text = Cleaner.remove_punctuation(text, less='.;:!?…')
		text = Cleaner.transform_punctuation(text)
		text = Cleaner.transform_numbers(text)
		text = Cleaner.trim(text)
		return text


class ParaSeuFilhoLerDataSet(DataSet):
	def _clean_text_file(self, text):
		text = Cleaner.lowercase(text)
		text = Cleaner.remove_newlines(text)
		text = fix(text)
		text = Cleaner.remove_punctuation(text, less='.;:!?…')
		text = Cleaner.transform_punctuation(text)
		text = Cleaner.transform_numbers(text)
		text = Cleaner.trim(text)
		return text


class CHCDataSet(DataSet):
	def _clean_text_file(self, text):
		text = Cleaner.lowercase(text)
		text = Cleaner.remove_newlines(text)
		text = Cleaner.remove_tags(['a', 'author', 'date', 'em', 'image', 'span', 'subtitle', 'title', 'url'], text)
		text = fix(text)
		text = Cleaner.remove_punctuation(text, less='.;:!?…')
		text = Cleaner.transform_punctuation(text)
		text = Cleaner.transform_numbers(text)
		text = Cleaner.trim(text)
		return text


class LivrosDidaticosDataSet(DataSet):
	def _clean_text_file(self, text):
		text = Cleaner.lowercase(text)
		text = Cleaner.remove_newlines(text)
		text = Cleaner.remove_tags(['a', 'author', 'date', 'em', 'image', 'span', 'subtitle', 'title', 'url'], text)
		text = fix(text)
		text = Cleaner.remove_punctuation(text, less='.;:!?…')
		text = Cleaner.transform_punctuation(text)
		text = Cleaner.transform_numbers(text)
		text = Cleaner.trim(text)
		return text


class MundoEstranhoDataSet(DataSet):
	def _clean_text_file(self, text):
		text = Cleaner.lowercase(text)
		text = Cleaner.remove_newlines(text)
		text = Cleaner.remove_tags(['a', 'author', 'date', 'em', 'image', 'span', 'subtitle', 'title', 'url'], text)
		text = fix(text)
		text = Cleaner.remove_punctuation(text, less='.;:!?…')
		text = Cleaner.transform_punctuation(text)
		text = Cleaner.transform_numbers(text)
		text = Cleaner.trim(text)
		return text