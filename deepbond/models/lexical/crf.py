import numpy as np
import logging
from pprint import pformat
from deepbond.models.lexical import LexicalModel

logger = logging.getLogger(__name__)


class CRF(LexicalModel):

	def _prepare_params(self):
		pass

	def build(self, algorithm='lbfgs', min_freq=20, verbose=True):
		from sklearn_crfsuite import CRF as CRFSuite
		# all_possible_transitions=False
		self.classifier = CRFSuite(algorithm=algorithm, min_freq=min_freq)

	def prepare_input(self, dataset, strategy):
		X = []
		X_text, gold = dataset.as_matrix(ids=False)
		X_text = strategy.prepare_input(X_text)
		X.append(X_text)

		X_POS = self.features.get_POS(X_text)
		X_POS = strategy.prepare_input(X_POS)
		X.append(X_POS)
		
		for a,b in zip(X_POS, X_text):
			if len(a) != len(b):
				print(len(a), len(b))
				print(a)
				print(dataset.indexes_to_words([b]))
		
		Y = strategy.prepare_output(gold)
		X = self._transform_text_input_to_dict(X)
		return X, Y, gold

	def _text_features(self, sent_t, sent_p, i):
		features = {}
		word = sent_t[i]
		pos = sent_p[i]
		word_prev 	= '<BOS>' if i == 0 			else sent_t[i-1]
		word_next 	= '<EOS>' if i == len(sent_t)-1 else sent_t[i+1]
		pos_prev 	= '<BOS>' if i == 0 			else sent_p[i-1]
		pos_next 	= '<EOS>' if i == len(sent_p)-1 else sent_p[i+1]
		features['bias'] 		= 1.0
		features['word_prev'] 	= word_prev
		features['word'] 		= word
		features['word_next'] 	= word_next
		if self.use_pos:
			features['pos_prev'] = pos_prev
			features['pos'] 	 = pos
			features['pos_next'] = pos_next
		
		if self.use_embeddings:
			for i, v in enumerate(self.features.embeddings.get_vector(word)):
				features['emb_%d' % i] = v
		
		if self.use_handcrafted:
			n = len(sent_t)
			m = len(sent_p)
			# assert(n==m)

			features['dup_single_m4'] = int(sent_t[i-4] == word) if i-4 >= 0 else 0
			features['dup_single_m3'] = int(sent_t[i-3] == word) if i-3 >= 0 else 0
			features['dup_single_m2'] = int(sent_t[i-2] == word) if i-2 >= 0 else 0
			features['dup_single_m1'] = int(sent_t[i-1] == word) if i-1 >= 0 else 0
			features['dup_single_p1'] = int(sent_t[i+1] == word) if i+1 < n else 0
			features['dup_single_p2'] = int(sent_t[i+2] == word) if i+2 < n else 0
			features['dup_single_p3'] = int(sent_t[i+3] == word) if i+3 < n else 0
			features['dup_single_p4'] = int(sent_t[i+4] == word) if i+4 < n else 0

			features['dup_pos_single_m4'] = int(sent_p[i-4] == pos) if i-4 >= 0 else 0
			features['dup_pos_single_m3'] = int(sent_p[i-3] == pos) if i-3 >= 0 else 0
			features['dup_pos_single_m2'] = int(sent_p[i-2] == pos) if i-2 >= 0 else 0
			features['dup_pos_single_m1'] = int(sent_p[i-1] == pos) if i-1 >= 0 else 0
			features['dup_pos_single_p1'] = int(sent_p[i+1] == pos) if i+1 < m else 0
			features['dup_pos_single_p2'] = int(sent_p[i+2] == pos) if i+2 < m else 0
			features['dup_pos_single_p3'] = int(sent_p[i+3] == pos) if i+3 < m else 0
			features['dup_pos_single_p4'] = int(sent_p[i+4] == pos) if i+4 < m else 0

			features['dup_pair_m4'] = int((sent_t[i-4], sent_t[i-3]) == (sent_t[i], sent_t[i+1])) if i-4 >= 0 and i+1 < n else 0
			features['dup_pair_m3'] = int((sent_t[i-3], sent_t[i-2]) == (sent_t[i], sent_t[i+1])) if i-3 >= 0 and i+1 < n else 0
			features['dup_pair_m2'] = int((sent_t[i-2], sent_t[i-1]) == (sent_t[i], sent_t[i+1])) if i-2 >= 0 and i+1 < n else 0
			features['dup_pair_m1'] = int((sent_t[i-1], sent_t[i-0]) == (sent_t[i], sent_t[i+1])) if i-1 >= 0 and i+1 < n else 0
			features['dup_pair_p1'] = int((sent_t[i+0], sent_t[i+1]) == (sent_t[i], sent_t[i+1])) if i+1 < n else 0
			features['dup_pair_p2'] = int((sent_t[i+1], sent_t[i+2]) == (sent_t[i], sent_t[i+1])) if i+2 < n else 0
			features['dup_pair_p3'] = int((sent_t[i+2], sent_t[i+3]) == (sent_t[i], sent_t[i+1])) if i+3 < n else 0
			features['dup_pair_p4'] = int((sent_t[i+3], sent_t[i+4]) == (sent_t[i], sent_t[i+1])) if i+4 < n else 0

			features['dup_pos_pair_m4'] = int((sent_p[i-4], sent_p[i-3]) == (sent_p[i], sent_p[i+1])) if i-4 >= 0 and i+1 < m else 0
			features['dup_pos_pair_m3'] = int((sent_p[i-3], sent_p[i-2]) == (sent_p[i], sent_p[i+1])) if i-3 >= 0 and i+1 < m else 0
			features['dup_pos_pair_m2'] = int((sent_p[i-2], sent_p[i-1]) == (sent_p[i], sent_p[i+1])) if i-2 >= 0 and i+1 < m else 0
			features['dup_pos_pair_m1'] = int((sent_p[i-1], sent_p[i-0]) == (sent_p[i], sent_p[i+1])) if i-1 >= 0 and i+1 < m else 0
			features['dup_pos_pair_p1'] = int((sent_p[i+0], sent_p[i+1]) == (sent_p[i], sent_p[i+1])) if i+1 < m else 0
			features['dup_pos_pair_p2'] = int((sent_p[i+1], sent_p[i+2]) == (sent_p[i], sent_p[i+1])) if i+2 < m else 0
			features['dup_pos_pair_p3'] = int((sent_p[i+2], sent_p[i+3]) == (sent_p[i], sent_p[i+1])) if i+3 < m else 0
			features['dup_pos_pair_p4'] = int((sent_p[i+3], sent_p[i+4]) == (sent_p[i], sent_p[i+1])) if i+4 < m else 0

			features['eos_m2'] = int(sent_t[i-2] == '.') if i-2 >= 0 else 0
			features['eos_m1'] = int(sent_t[i-1] == '.') if i-1 >= 0 else 0
			features['eos_p1'] = int(sent_t[i+1] == '.') if i+1 < n else 0
			features['eos_p2'] = int(sent_t[i+2] == '.') if i+2 < n else 0

			ext_p = lambda w, s: '<NONE>' if s > len(w) else w[:s]
			features['prefix_p1_s2'] = int(ext_p(sent_t[i+1], 2) == word) if i+1 < n else 0
			features['prefix_p1_s3'] = int(ext_p(sent_t[i+1], 3) == word) if i+1 < n else 0
			features['prefix_p1_s4'] = int(ext_p(sent_t[i+1], 4) == word) if i+1 < n else 0
			features['prefix_p2_s2'] = int(ext_p(sent_t[i+2], 2) == word) if i+2 < n else 0
			features['prefix_p2_s3'] = int(ext_p(sent_t[i+2], 3) == word) if i+2 < n else 0
			features['prefix_p2_s4'] = int(ext_p(sent_t[i+2], 4) == word) if i+2 < n else 0


		return features

	def _transform_text_input_to_dict(self, data):
		l = []
		text, pos = data
		for sent_t, sent_p in zip(text, pos):
			l.append([self._text_features(sent_t, sent_p, i) for i in range(len(sent_t))])
		return l


	def _audio_features(self, sent, i):
		features = {}
		duration = sent[i][0]
		energy = sent[i][1]
		pitch = sent[i][2]
		duration_prev 	= '<BOS>' if i == 0 			else sent[i-1][0]
		duration_next 	= '<EOS>' if i == len(sent)-1 	else sent[i+1][0]
		energy_prev		= '<BOS>' if i == 0 			else sent[i-1][1]
		energy_next	 	= '<EOS>' if i == len(sent)-1 	else sent[i+1][1]
		pitch_prev 		= '<BOS>' if i == 0 			else sent[i-1][2]
		pitch_next 		= '<EOS>' if i == len(sent)-1 	else sent[i+1][2]
		pause 			= 'no' if sent[i][3] == 0 else 'short' if sent[i][3] < 0.4 else 'long'
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


	def save_weights(self, filename):
		pass

	def load_weights(self, filename):
		pass
	