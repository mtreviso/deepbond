from abc import ABCMeta, abstractmethod
import numpy as np

class ProsodicModel(metaclass=ABCMeta):

	def __init__(self, vocabulary=None, features=None, input_length=None, nb_classes=2):
		'''
		:param vocabulary: a dict where keys are words
		:param features: a feature class instance
		:param input_length: sentence size, a value if fixed (padding) or None if variable (bucket)
		:param nb_classes: number of targets, usually max(y) + 1
		'''
		self.vocabulary = vocabulary
		self.features = features
		self.nb_classes = nb_classes
		self.input_length = input_length
		self.classifier = None
		self._prepare_params()
		self.name = 'prosodic'

	def prepare_input(self, dataset, strategy):
		X = []
		_, gold = dataset.as_matrix(ids=True)
		mask_lines = list(map(len, gold))
		# self.features.test_prosodic(dataset.pros_texts, dataset.word_texts)
		X_pros = self.features.get_prosodic(dataset.pros_texts, mask_lines=mask_lines)
		X.append(strategy.prepare_input(X_pros))
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
				output_data = {}
				input_data['input_pros'] = np.array([data_in[len(input_data)][i] for i in indexes])
				output_data['output_source'] = np.array([data_out[i] for i in indexes])
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
