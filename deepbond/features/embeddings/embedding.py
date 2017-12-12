from abc import ABCMeta, abstractmethod
import numpy as np


class Embedding(metaclass=ABCMeta):

	def __init__(self, lowercase=False, dimensions=50, min_count=5, workers=4, window_size=5, epochs=5, oov_word='*RARE*'):
		self.lowercase = lowercase
		self.dimensions = dimensions
		self.min_count = min_count
		self.workers = workers
		self.window_size = window_size
		self.epochs = epochs
		self.oov_word = oov_word
		self.oov = dict()
		self.model = dict()
		self.random_vector = None
		self._vocabulary = None

	@abstractmethod
	def load(self, load_file): 
		pass

	@abstractmethod
	def save(self, save_file): 
		pass

	@abstractmethod
	def train(self, train_file, **kwargs): 
		pass

	@property
	def vocabulary(self): 
		if self._vocabulary is None:
			self._vocabulary = dict(zip(self.model.keys(), range(len(self.model))))
		return self._vocabulary

	@property
	def vocabulary_size(self):
		return len(self.vocabulary)

	def get_vector(self, word):
		if word in self.model:
			return self.model[word]
		if self.lowercase and word.lower() in self.model:
			return self.model[word.lower()]
		if word not in self.oov:
			self.oov[word] = 0
		self.oov[word] += 1
		if self.oov_word != None and self.oov_word in self.model:
			return self.model[self.oov_word]
		if self.random_vector is None:
			self.random_vector = self.generate_random_vector()
		return self.random_vector

	def generate_random_vector(self):
		epsilon = np.sqrt(6) / np.sqrt(self.dimensions)
		return np.random.random(self.dimensions) * 2 * epsilon - epsilon

	def statistics(self, top_k=10):
		nb_oovs = len(self.oov)
		nb_occur_oovs = sum(list(self.oov.values()))
		top_k_oovs = sorted(list(self.oov.items()), key=lambda x: x[1], reverse=True)
		return nb_oovs, nb_occur_oovs, top_k_oovs[:top_k]

	def oov_statistics(self, list_of_words, top_k=10):
		oovs = {}
		for x in list_of_words:
			word = x.lower() if self.lowercase else x
			if word not in self.model:
				if word not in oovs:
					oovs[word] = 0
				oovs[word] += 1
		nb_oovs = len(oovs)
		nb_occur_oovs = sum(list(oovs.values()))
		top_k_oovs = sorted(list(oovs.items()), key=lambda x: x[1], reverse=True)
		return nb_oovs, nb_occur_oovs, top_k_oovs[:top_k]


class LoadableEmbedding(Embedding):

	def save(self, save_file):
		raise Exception('This model is already saved.')

	def train(self, train_file):
		raise Exception('This class just load word embeddings.')
