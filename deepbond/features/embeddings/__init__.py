from __future__ import absolute_import
from .fasttext import *
from .glove import *
from .loadable import *
from .wang2vec import *
from .word2vec import *

class AvailableEmbeddings:
	map_emb = {
		'word2vec': Word2Vec,
		'glove': 	Glove,
		'fasttext': FastText,
		'wang2vec': Wang2Vec,
		'random': 	RandomVectors,
		'text': 	TextVectors,
		'fonseca': 	FonsecaVectors,
		'senna': 	SennaVectors,
		'id': 		IdVectors
	}
	@staticmethod
	def get(word):
		return AvailableEmbeddings.map_emb[word]