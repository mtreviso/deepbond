import numpy as np
from deepbond.models.strategies import Strategy
from deepbond.utils import pad_sequences, pad_sequences_3d, unpad_sequences, vectorize

class PaddingStrategy(Strategy):

	def prepare_input(self, sequence):
		if isinstance(sequences[0], (np.ndarray, list)):
			return pad_sequences_3d(sequence, maxlen=self.input_length, mask_value=0)
		return pad_sequences(sequence, maxlen=self.input_length, mask_value=0)

	def unprepare(self, sequence, map_with=None):
		return unpad_sequences(sequence, map_with=map_with)

	def prepare_output(self, sequence, one_hot_dim=None):
		return vectorize(self.prepare_input(sequence), one_hot_dim=one_hot_dim)

