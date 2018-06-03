import pickle
import glove  # pip install git+https://github.com/maciejkula/glove-python.git
from .embedding import Embedding

class DGlove(glove.Glove):
	
	@classmethod
	def load(cls, filename):
		instance = DGlove()
		with open(filename, 'rb') as savefile:
			instance.__dict__ = pickle.load(savefile)
		return instance

	def __contains__(self, x):
		return x in self.dictionary

	def __getitem__(self, x):
		return self.word_vectors[self.dictionary[x]]


class Glove(Embedding):

	@property
	def vocabulary(self): 
		if self._vocabulary is None:
			self._vocabulary = self.model.dictionary
		return self._vocabulary

	def load(self, load_file):
		self.model = DGlove.load(load_file)
		self.dimensions = self.model.no_components

	def save(self, save_file):
		self.model.save(save_file)

	def _read_corpus(self, filename):
		with open(filename, 'r') as datafile:
			for line in datafile:
				yield line.strip().split()

	def train(self, train_file, **kwargs):
		corpus_model = glove.Corpus()
		corpus_model.fit(self._read_corpus(train_file), window=self.window_size)
		self.model = DGlove(no_components=self.dimensions, **kwargs)
		self.model.fit(corpus_model.matrix, no_threads=self.workers, epochs=self.epochs, verbose=True)
		self.model.add_dictionary(corpus_model.dictionary)
