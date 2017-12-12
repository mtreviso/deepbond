import numpy as np
import logging

from sklearn_crfsuite import CRF

from deepbond.utils import unroll, unvectorize
from deepbond.statistics import Statistics

logger = logging.getLogger(__name__)


class WindowEnsembleCRF:
	
	def __init__(self, features=None, nb_classes=2, input_length=3, use_embeddings=True, use_pos=True):
		self.features = features
		self.nb_classes = nb_classes
		self.input_length = input_length
		self.use_embeddings = use_embeddings
		self.use_pos = use_pos

	def build(self):
		logger.info('Building...')
		pass

	def _prepare_input(self, ds, mask_prosody=True):
		X_text, Y = ds.as_matrix(ids=False)
		X_POS = self.features.get_POS(ds.indexes_to_words(X_text))
		X_mask_lines =  None if not mask_prosody else list(map(len, X_text))
		X_pros = self.features.get_prosodic(ds.pros_texts, mask_lines=X_mask_lines, average=True)
		text_data  = [[X_text, X_POS], Y]
		audio_data = [X_pros, Y]
		return text_data, audio_data

	def _text_features(self, sent_t, sent_p, i):
		features = {}
		word = sent_t[i]
		pos = sent_p[i]
		word_prev 	= '<BOS>' if i == 0 			else sent_t[i-1]
		word_next 	= '<EOS>' if i == len(sent_t)-1 else sent_t[i+1]
		pos_prev 	= '<BOS>' if i == 0 			else sent_p[i-1]
		pos_next 	= '<EOS>' if i == len(sent_p)-1 else sent_p[i+1]
		features['bias'] = 1.0
		features['word_prev'] = word_prev
		features['word'] = word
		features['word_next'] = word_next
		if self.use_pos:
			features['pos_prev'] = pos_prev
			features['pos'] = pos
			features['pos_next'] = pos_next
		if self.use_embeddings:
			for i, v in enumerate(self.features.embeddings.get_vector(word)):
				features['emb_%d' % i] = v
		return features

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
		pause = 'no' if sent[i][3] == 0 else 'short' if sent[i][3] < 0.4 else 'long'
		features['bias'] = 1.0
		features['duration_prev'] = duration_prev
		features['duration'] = duration
		features['duration_next'] = duration_next
		features['energy_prev'] = energy_prev
		features['energy'] = energy
		features['energy_next'] = energy_next
		features['pitch_prev'] = pitch_prev
		features['pitch'] = pitch
		features['pitch_next'] = pitch_next
		features['pause'] = pause
		return features

	def _transform_output_to_dict(self, data, map_with=None):
		if map_with is not None:
			return [list(map(str, data[i])) for i, sent in enumerate(map_with) if sent is not None]
		return list(map(lambda y: list(map(str, y)), data))

	def _untransform_output(self, data):
		return list(map(lambda y: list(map(int, y)), data))

	def _transform_text_input_to_dict(self, data):
		l = []
		text, pos = data
		for sent_t, sent_p in zip(text, pos):
			l.append([self._text_features(sent_t, sent_p, i) for i in range(len(sent_t))])
		return l

	def _transform_audio_input_to_dict(self, data):
		l = []
		for sent in data:
			if sent is not None:
				l.append([self._audio_features(sent, i) for i in range(len(sent))])
		return l

	def _argmax_output_dict(self, data):
		l = []
		for sent in data:
			for w in sent:
				l.append([w['0'], w['1']])
		return l

	def train(self, ds_train, ds_test, nb_epoch=30):

		print('Prepraing input...')
		text_train_data, audio_train_data = self._prepare_input(ds_train, mask_prosody=False)
		text_test_data, audio_test_data = self._prepare_input(ds_test, mask_prosody=False)

		X_train_text, Y_train_text = text_train_data
		X_train_audio, Y_train_audio = audio_train_data
		X_test_text, Y_test_text = text_test_data
		X_test_audio, Y_test_audio = audio_test_data

		# http://sklearn-crfsuite.readthedocs.io/en/latest/api.html
		text_model = CRF(algorithm='lbfgs', min_freq=20)
		audio_model = CRF(algorithm='lbfgs', min_freq=20)

		logger.info('Training text %s...\n' % str(text_model))
		X_train = self._transform_text_input_to_dict(X_train_text)
		Y_train = self._transform_output_to_dict(Y_train_text)
		text_model.fit(X_train, Y_train)
		P_train = text_model.predict(X_train)
		Statistics.print_metrics(unroll(Y_train_text), unroll(self._untransform_output(P_train)))


		logger.info('Training audio %s...\n' % str(audio_model))
		X_train = self._transform_audio_input_to_dict(X_train_audio)
		Y_train = self._transform_output_to_dict(Y_train_audio, map_with=X_train_audio)
		audio_model.fit(X_train, Y_train)
		P_train = audio_model.predict(X_train)
		Statistics.print_metrics(unroll(self._untransform_output(Y_train)), unroll(self._untransform_output(P_train)))


		logger.info('Testing...')
		pred_t = text_model.predict(self._transform_text_input_to_dict(X_test_text))
		Statistics.print_metrics(unroll(Y_test_text), unroll(self._untransform_output(pred_t)))
		text_metrics = list(Statistics.get_metrics(unroll(Y_test_text), unroll(self._untransform_output(pred_t))))

		pred_a = audio_model.predict(self._transform_audio_input_to_dict(X_test_audio))
		Statistics.print_metrics(unroll(Y_test_audio), unroll(self._untransform_output(pred_a)))
		audio_metrics = list(Statistics.get_metrics(unroll(Y_test_audio), unroll(self._untransform_output(pred_a))))


		logger.info('Evaluating...')
		pred_t = text_model.predict_marginals(self._transform_text_input_to_dict(X_test_text))
		pred_t = np.array(self._argmax_output_dict(pred_t))
		
		pred_a = audio_model.predict_marginals(self._transform_audio_input_to_dict(X_test_audio))
		pred_a = np.array(self._argmax_output_dict(pred_a))
		
		gold = np.array(unroll(Y_test_audio))
		total_metrics = [0, 0, 0]
		partitions = list(map(lambda p: p/10.0, range(0, 11)))
		max_p = 0
		Y_pred = []

		for p in partitions:
			logger.info('F1 for p: %.2f' % p)
			pred = p*pred_t + (1-p)*pred_a
			f1 = Statistics.print_metrics(gold, unvectorize(pred))
			if f1 > total_metrics[0]:
				total_metrics = list(Statistics.get_metrics(gold, unvectorize(pred)))
				max_p = p
				Y_pred = unvectorize(pred)

		return np.array(text_metrics), np.array(audio_metrics), np.array(total_metrics), max_p, Y_pred

