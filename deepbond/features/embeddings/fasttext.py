import fasttext # pip install git+https://github.com/salestock/fastText.py
from .embedding import Embedding
import os

class FastText(Embedding):

	@property
	def vocabulary(self): 
		if self._vocabulary is None:
			self._vocabulary = self.model.words
		return self._vocabulary

	def load(self, load_file):
		self.model = fasttext.load_model(load_file)
		self.dimensions = self.model.dim

	def save(self, save_file):
		os.rename('model_fasttext', save_file)

	def train(self, train_file, **kwargs):
		self.model = fasttext.skipgram(train_file, 'model_fasttext', dim=self.dimensions, min_count=self.min_count, 
										thread=self.workers, ws=self.window_size, epoch=self.epochs, **kwargs)

	def get_vector(self, word):
		if word in self.model:
			return self.model[word]
		if self.lowercase and word.lower() in self.model:
			return self.model[word.lower()]
		if word not in self.oov:
			self.oov[word] = 0
		self.oov[word] += 1
		return self.model[word]
