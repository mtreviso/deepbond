import numpy as np
import logging
from pprint import pformat
from deepbond.models.prosodic import ProsodicModel

logger = logging.getLogger(__name__)


class CRF(ProsodicModel):

	def _prepare_params(self):
		pass

	def build(self, algorithm='lbfgs', min_freq=20, verbose=True):
		from sklearn_crfsuite import CRF as CRFSuite
		# all_possible_transitions=False
		self.classifier = CRFSuite(algorithm=algorithm, min_freq=min_freq)

	def prepare_input(self, dataset, strategy):
		X = []
		_, gold = dataset.as_matrix(ids=True)
		mask_lines = list(map(len, gold))
		X_pros = self.features.get_prosodic(dataset.pros_texts, mask_lines=mask_lines, average=True)
		Y = strategy.prepare_output(gold)
		X = self._transform_audio_input_to_dict(X_pros)
		return X, Y, gold

	def _audio_features(self, sent, i):
		features = {}
		duration = sent[i][-3]
		energy = sent[i][-2]
		pitch = sent[i][-1]
		duration_prev 	= '<BOS>' if i == 0 			else sent[i-1][-3]
		duration_next 	= '<EOS>' if i == len(sent)-1 	else sent[i+1][-3]
		energy_prev		= '<BOS>' if i == 0 			else sent[i-1][-2]
		energy_next	 	= '<EOS>' if i == len(sent)-1 	else sent[i+1][-2]
		pitch_prev 		= '<BOS>' if i == 0 			else sent[i-1][-1]
		pitch_next 		= '<EOS>' if i == len(sent)-1 	else sent[i+1][-1]
		pause 			= 'no' if sent[i][0] == 0 else 'short' if sent[i][0] < 0.4 else 'long'
		features['bias'] 			= 1.0
		features['duration_prev'] 	= duration_prev
		features['duration'] 		= duration
		features['duration_next'] 	= duration_next
		features['energy_prev'] 	= energy_prev
		features['energy'] 			= energy
		features['energy_next'] 	= energy_next
		features['pitch_prev'] 		= pitch_prev
		features['pitch'] 			= pitch
		features['pitch_next'] 		= pitch_next
		features['pause'] 			= pause
		return features

	def _transform_audio_input_to_dict(self, data):
		l = []
		for sent in data:
			if sent is not None:
				l.append([self._audio_features(sent, i) for i in range(len(sent))])
		return l

	def save_weights(self, filename):
		pass

	def load_weights(self, filename):
		pass
	