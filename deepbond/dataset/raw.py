from deepbond.dataset import DataSet
from deepbond.helpers import Cleaner
from deepbond.dataset import SentTokenizer
import re

tokenizer = SentTokenizer() # vocabulary

class RawDataSet(DataSet):

	def __init__(self, texts=[], audios=[], vocabulary=None):
		"""
		Class to deal with an annotated corpus
		:param texts: list of raw texts as strings
		:param audios: list of paths to a prosodic file (.csv)
		---
		The nth text and the nth audio should be mutual (same transcript)
		"""
		if len(texts) == 0 and len(audios) == 0:
			raise Exception('You should pass a list of texts or a list of audios!')

		if len(texts) > 0 and len(audios) > 0 and len(texts) != len(audios):
			raise Exception('The number of the texts audios should be equal!')

		if len(texts) > 0 and not isinstance(texts[0], str):
			raise Exception('`texts` should be a list of strings')

		if len(audios) > 0 and not isinstance(audios[0], str):
			raise Exception('`audios` should be a list of paths to a prosodic file (.csv)')

		# if we have a saved vocab, then the tokenizer will load it
		if vocabulary is not None and not tokenizer.loaded:
			tokenizer.load_vocabulary(vocabulary)
		
		self.texts = []
		self.word_texts = []
		self.word_id_texts = []
		self.pros_texts = []
		
		if len(texts) > 0:
			self._read_raw_texts(texts)
			self._fit_tokenizer(build_vocab=False)
			self._set_nb_attributes()
		
		if len(audios) > 0:
			self._read_raw_audios(audios)

	def _read_raw_texts(self, texts, min_sentence_size=5):
		for txt in texts:
			txt = self._clean_text_file(txt)
			tks = txt.strip().split()
			if txt.strip() and len(tks) >= min_sentence_size and tks[0] not in '.!?:;':
				self.texts.append(txt)
			else:
				raise Exception('Text too short or begin with a punctuation:\n %s' % txt)

	def _read_raw_audios(self, audios):
		import pandas as pd
		dfs_pros = []
		for filename in audios:
			pros = pd.read_csv(filename, encoding='utf8')
			pros = pros.drop(['n interval', 'Start', 'Finish', 'fo1', 'fo3'], axis=1)
			pros = pros.replace('--undefined--', 0)
			pros = pros.where((pd.notnull(pros)), None) # transform NaN to None
			# normalize per speaker too
			scale_minmax = lambda x: (x - x.min()) / (x.max() - x.min())
			pros['Intensity'] = scale_minmax(pros['Intensity'].astype(float))
			pros['fo2'] = scale_minmax(pros['fo2'].astype(float))
			pros['Duration'] = pros['Duration'] / 1000
			dfs_pros.append(pros)
		self.pros_texts = pd.concat(dfs_pros)
		self.pros_texts = list(self.pros_texts.groupby('Filename'))

	def _clean_text_file(self, text):
		# since deepbond works with transcript we remove all punctuation from text
		text = Cleaner.remove_punctuation(text, less='.-')
		text = Cleaner.lowercase(text)
		text = Cleaner.remove_newlines(text)
		text = Cleaner.trim(text)
		text = re.sub(r'(\S+)([\*\.])', r'\1 \2', text.strip())
		text = re.sub(r'([\*\.])(\S+)', r'\1 \2', text.strip())
		return text.strip()

