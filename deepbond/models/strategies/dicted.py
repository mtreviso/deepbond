import numpy as np
from deepbond.models.strategies import Strategy
from deepbond.utils import vectorize, unvectorize

class DictedStrategy(Strategy):

	def prepare_input(self, sequence):
		return sequence

	def unprepare(self, sequence):
		return np.array(list(map(lambda y: list(map(int, y)), sequence)))

	def prepare_output(self, sequence, one_hot_dim=None):
		return list(map(lambda y: list(map(str, y)), sequence))
