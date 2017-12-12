from .embedding import LoadableEmbedding
import numpy as np

class FonsecaVectors(LoadableEmbedding):
	"""
	A class that just load the vectors in specified file and
	return a np.array with shape (vocabulary_size, word_embedding_dimension)
	see: http://nilc.icmc.usp.br/nlpnet/models.html
	---
	Attention: the vocabulary is in lowercase
	"""

	def load(self, load_file):
		slash = '/' if load_file[-1] != '/' else ''
		weights_filename = load_file + slash + 'types-features.npy'
		vocabulary_filename = load_file + slash + 'vocabulary.txt'
		weights = np.load(weights_filename)
		vocab = [line.strip() for line in open(vocabulary_filename, 'r', encoding='utf-8')]
		self.model = dict(zip(vocab, weights))
		self.dimensions = len(weights[0])
		self.oov_word = '*rare*'
		self.lowercase = True


class RandomVectors(LoadableEmbedding):

	def load(self):
		self.model = dict()

	def get_vector(self, word):
		if word not in self.model:
			self.model[word] = self.generate_random_vector()
		return self.model[word]


class TextVectors(LoadableEmbedding):
	"""
	A simples class that just load the vectors from
	http://nlp.stanford.edu/projects/glove/
	"""

	def load(self, load_file):
		self.model = dict()
		word_dim = 0
		with open(load_file, 'r', encoding="utf-8") as f:
			for line in f:
				data = line.split()
				self.model[data[0]] = list(map(float, data[1:]))
				word_dim = len(data)-1
		self.dimensions = word_dim


class SennaVectors(LoadableEmbedding):

	def load(self, load_file):
		self.model = dict()
		word_dim = 0
		slash = '/' if load_file[-1] != '/' else ''
		weights_filename = load_file + slash + 'embeddings.txt'
		vocabulary_filename = load_file + slash + 'words.lst'

		weights, vocab = [], []
		with open(weights_filename, 'r', encoding="utf-8") as f:
			for line in f:
				weights.append(list(map(float, line.split())))

		with open(vocabulary_filename, 'r', encoding="utf-8") as f:
			for line in f:
				vocab.append(line.strip())

		self.model = dict(zip(vocab, weights))
		self.dimensions = len(weights[0])

class IdVectors(LoadableEmbedding):
	def load(self, vocab):
		self.dimensions = 1
		self.model = dict(zip(vocab.keys(), map(lambda x:[x+1], range(len(vocab)))))
		self.random_vector = [len(vocab)+1]
