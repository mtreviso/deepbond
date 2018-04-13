from abc import ABCMeta, abstractmethod
import numpy as np

class LexicalModel(metaclass=ABCMeta):

	def __init__(self, vocabulary=None, features=None, input_length=None, nb_classes=2, use_embeddings=True, use_pos=True, use_handcrafted=False):
		'''
		:param vocabulary: a dict where keys are words
		:param features: a feature class instance
		:param input_length: sentence size, a value if fixed (padding), or None if variable
		:param nb_classes: number of targets, usually max(y) + 1
		'''
		self.vocabulary = vocabulary
		self.features = features
		self.nb_classes = nb_classes
		self.input_length = input_length
		self.use_embeddings = use_embeddings
		self.use_pos = use_pos
		self.use_handcrafted = use_handcrafted
		self.classifier = None
		self._prepare_params()
		self.name = 'lexical'

	def prepare_input(self, dataset, strategy):
		X, X_ind, X_ind_POS = [], [], []
		X_indexes, gold = dataset.as_matrix(ids=True)
		if self.use_embeddings:
			X_ind = strategy.prepare_input(X_indexes)
			X.append(X_ind)
		if self.use_pos:
			X_ind_POS = self.features.get_POS(dataset.indexes_to_words(X_indexes))
			X_POS = strategy.prepare_input(X_ind_POS)
			X.append(X_POS)
		if self.use_handcrafted:
			X_hc = self.features.get_handcrafted(dataset.indexes_to_words(X_indexes), X_ind_POS)
			X.append(strategy.prepare_input(X_hc))
		
		Y = strategy.prepare_output(gold, one_hot_dim=self.nb_classes)
		return X, Y, gold

	def get_batch(self, data_in, data_out, sample_weights, lengths, data_by_length, shuffle=True, kind='train'):
		while True: # a new epoch
			if shuffle:
				np.random.shuffle(lengths)
			for length in lengths:
				indexes = data_by_length[length]
				if shuffle:
					np.random.shuffle(indexes)
				input_data = {}
				if self.use_embeddings:
					input_data['input_emb'] 	= np.array([data_in[len(input_data)][i] for i in indexes])
				if self.use_pos:
					input_data['input_pos'] 	= np.array([data_in[len(input_data)][i] for i in indexes])
				if self.use_handcrafted:
					input_data['input_hc']		= np.array([data_in[len(input_data)][i] for i in indexes])
				if len(input_data) == 0:
					raise Exception('You must use at least one feature.')
				output_data = {'output_source': np.array([data_out[i] for i in indexes])}
				sw_data = None if sample_weights is None else np.array([sample_weights[i] for i in indexes])
				yield (input_data, output_data, sw_data)
			if kind == 'predict':
				break

	@abstractmethod
	def _prepare_params(self):
		pass

	@abstractmethod
	def build(self, **kwargs):
		pass

	def save_weights(self, filename):
		self.classifier.save_weights(filename)

	def load_weights(self, filename):
		self.classifier.load_weights(filename)
