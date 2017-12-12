import numpy as np
from deepbond.models.strategies import Strategy
from deepbond.utils import vectorize, unvectorize

class BucketStrategy(Strategy):

	def prepare_input(self, sequence):
		return np.array(sequence)

	def unprepare(self, sequence):
		return sequence

	def prepare_output(self, sequence, one_hot_dim=None):
		return np.array([vectorize(np.array(y), one_hot_dim=one_hot_dim) for y in sequence])
