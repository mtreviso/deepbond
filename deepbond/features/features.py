import logging
import numpy as np

from deepbond.dataset import SentTokenizer
from deepbond.features import AvailableEmbeddings, POS, Prosodic, HandCrafted

logger = logging.getLogger(__name__)
tokenizer = SentTokenizer()  # vocabulary


class Features:

	def __init__(self, embedding_type='random', embedding_file=None,
				 POS_type='nlpnet', POS_file=None,
				 prosodic_type='principal', prosodic_classify=True, 
				 use_pos=True, use_embeddings=True, use_handcrafted=False):

		# POS
		self.use_pos = use_pos
		self.pos = POS(POS_type)
		if self.use_pos:
			self.pos.load(POS_file)

		# EMBEDDINGS
		self.use_embeddings = use_embeddings
		self.embedding_type = embedding_type
		self.embedding_file = embedding_file
		self.embeddings = AvailableEmbeddings.get(embedding_type)()
		self.embeddings.load(tokenizer.word_index if embedding_type == 'id' else embedding_file)

		# PROSODIC
		self.prosodic = Prosodic(prosodic_type, prosodic_classify,
								 nb_features=3, first=1, last=3, max_size=10, pad_value=-1)

		# HANDCRAFTED
		self.use_handcrafted = use_handcrafted
		if self.use_handcrafted:
			self.handcrafted = HandCrafted(wang_single_limit=4, wang_pair_limit=4, 
										prefix_size=(2,4), prefix_limit=2, eos_limit=2, use_pos=use_pos)


	def save(self, filename):
		import json
		data = {
			'use_pos': self.use_pos,
			'use_embeddings': self.use_embeddings,
			'use_handcrafted': self.use_handcrafted,
			'POS_type': self.pos.type,
			'POS_file': self.pos.filename,
			'embedding_type': self.embedding_type,
			'embedding_file': self.embedding_file,
			'prosodic_type': self.prosodic.type,
			'prosodic_classify': self.prosodic.classify
		}
		with open(filename, 'w') as f:
			json.dump(data, f)
	
	def load(self, filename):
		import json
		with open(filename, 'r') as f:
			data = json.load(f)
		self.__init__(**data)

	def info(self):
		if self.use_embeddings:
			logger.info('Embeddings type: {}'.format(type(self.embeddings).__name__))
			logger.info('Embeddings dim: {}'.format(self.embeddings.dimensions))
			logger.info('Embeddings vocab size: {}'.format(len(self.embeddings.vocabulary)))
		if self.use_pos:
			logger.info('POS type: {}'.format(self.pos.type))
			logger.info('POS vocab size: {}'.format(len(self.pos.vocabulary)))
		logger.info('Prosodic type: {}'.format(self.prosodic.type))
		logger.info('Prosodic classify: {}'.format(self.prosodic.classify))
		logger.info('Prosodic nb features: {}'.format(self.prosodic.nb_features))
		logger.info('Prosodic first: {}'.format(self.prosodic.nb_first))
		logger.info('Prosodic last: {}'.format(self.prosodic.nb_last))

	def get_embeddings(self, vocabulary):
		vocab_size = max(vocabulary.values()) + 1
		weights = np.random.randn(vocab_size, self.embeddings.dimensions)
		for word, index in vocabulary.items():
			weights[index] = self.embeddings.get_vector(word)
		return weights

	def get_handcrafted(self, word_texts, pos_texts):
		return self.handcrafted.get(word_texts, pos_texts)

	def embeddings_statistics(self, word_texts):
		if self.use_embeddings:
			logger.info('Top 10 embeddings misses in dataset:')
			words = [w_ for w in word_texts for w_ in w]
			nb_oovs, nb_occur_oovs, top_k_oovs = self.embeddings.oov_statistics(
				words)
			logger.info('Total de palavras fora do vocabulario: %d' % nb_oovs)
			logger.info(
				'Total de ocorrencia de palavras fora do vocabulario: %d' % nb_occur_oovs)
			for w, c in top_k_oovs:
				logger.info('%s: %d' % (w, c))

	def get_POS(self, texts):
		return self.pos.get(texts)

	def get_prosodic(self, pros_texts, mask_lines=None, mask_value=0.0, average=False, normalize=False):
		return self.prosodic.get(pros_texts, mask_lines=mask_lines,
		mask_value=mask_value, average=average, normalize=normalize)

	def test_prosodic(self, pros_texts, map_with):
		return self.prosodic.test_prosodic(pros_texts, map_with)
