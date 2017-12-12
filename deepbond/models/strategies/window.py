import numpy as np
from deepbond.models.strategies import Strategy
from deepbond.utils import unroll, convolve_sequences, convolve_sequences_3d, unconvolve_sequences, vectorize

class WindowStrategy(Strategy):

	def prepare_input(self, sequence):
		if isinstance(sequence[0][0], (np.ndarray, list)):
			return convolve_sequences_3d(sequence, self.input_length, left_pad_value=0, right_pad_value=0)
		return convolve_sequences(sequence, self.input_length, left_pad_value=0, right_pad_value=0)

	def unprepare(self, sequence):
		return sequence

	def prepare_output(self, sequence, one_hot_dim=None):
		return vectorize(np.array(unroll(sequence)), one_hot_dim=one_hot_dim)
