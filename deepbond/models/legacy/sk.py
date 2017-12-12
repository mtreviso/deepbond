import numpy as np
import logging

from sklearn.utils import compute_class_weight
from sklearn.preprocessing import scale
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
# from xgboost.sklearn import XGBClassifier

from deepbond.utils import unroll, vectorize, unvectorize, row_matrix, column_matrix, convolve_sequences, convolve_sequences_3d
from deepbond.statistics import Statistics

logger = logging.getLogger(__name__)


class WindowEnsembleSkLearn:
	
	def __init__(self, vocabulary, features=None, nb_classes=2, input_length=None, which='svm', use_embeddings=True, use_pos=True):
		self.vocabulary = vocabulary
		self.features = features
		self.nb_classes = nb_classes
		self.input_length = input_length
		self.which = which
		self.use_embeddings = use_embeddings
		self.use_pos = use_pos
		self.embeddings_weights = np.array(self.features.get_embeddings(self.vocabulary))
		self.POS_weights = np.random.random(size=(max(self.features.pos.vocabulary.values())+10, 10))

	def build(self):
		pass

	def _select_algorithm(self, cw_text=None, cw_audio=None):
		logger.info('Building...')
		self.text_model, self.audio_model = None, None
		if self.which == 'xgboost':
			self.text_model = XGBClassifier(n_estimators=200, max_depth=5)
			self.audio_model = XGBClassifier(n_estimators=200, max_depth=5)
		elif self.which == 'svm':
			self.text_model = svm.SVC(probability=True, class_weight=cw_text, C=100)
			self.audio_model = svm.SVC(probability=True, class_weight=cw_audio, C=100)
		elif self.which == 'extra_trees':
			self.text_model = ExtraTreesClassifier(n_estimators=200, criterion='entropy', class_weight=cw_text, max_depth=5, n_jobs=-1)
			self.audio_model = ExtraTreesClassifier(n_estimators=200, criterion='entropy', class_weight=cw_audio, max_depth=5, n_jobs=-1)
		elif self.which == 'decision_trees':
			self.text_model = DecisionTreeClassifier(criterion='entropy', class_weight=cw_text, max_depth=5)
			self.audio_model = DecisionTreeClassifier(criterion='entropy', class_weight=cw_audio, max_depth=5)
		elif self.which == 'gaussian_nb':
			self.text_model = GaussianNB()
			self.audio_model = GaussianNB()
		elif self.which == 'crf':
			raise Exception('Please, use the CRF model in crf.py file')
		else:
			raise Exception('Classifier not implemented')

	def _prepare_text_input(self, ds, valid_indexes=None):
		X_text, Y = ds.as_matrix(ids=True)
		if valid_indexes is not None:
			X_text = [X_text[i] for i in valid_indexes]
			Y = [Y[i] for i in valid_indexes]
		X_POS = self.features.get_POS(ds.indexes_to_words(X_text))
		X_text = convolve_sequences(X_text, self.input_length, left_pad_value=0, right_pad_value=0)
		X_text = np.array(list(map(lambda x: unroll(list(map(self.embeddings_weights.__getitem__, x))), X_text)))
		X_POS = convolve_sequences(X_POS, self.input_length, left_pad_value=0, right_pad_value=0)
		X_POS = np.array(list(map(lambda x: unroll(list(map(self.POS_weights.__getitem__, x))), X_POS)))
		X = [X_text, X_POS/(max(self.features.pos.vocabulary.values())+1)]
		Y = vectorize(column_matrix(unroll(Y), dtype=np.int32), one_hot_dim=self.nb_classes)
		Y = Y.reshape(Y.shape[0], Y.shape[2])
		return X, Y

	def _prepare_audio_input(self, ds, return_indexes=False):
		_, Y = ds.as_matrix()
		X_audio = self.features.get_prosodic(ds.pros_texts, mask_lines=None)
		valid_indexes = [i for i, x in enumerate(X_audio) if x != None]
		X_audio = [X_audio[i] for i in valid_indexes]
		X_audio = convolve_sequences_3d(X_audio, self.input_length, left_pad_value=0, right_pad_value=0)
		X = [X_audio]
		Y = vectorize(column_matrix(unroll([Y[i] for i in valid_indexes]), dtype=np.int32), one_hot_dim=self.nb_classes)
		Y = Y.reshape(Y.shape[0], Y.shape[2])
		if not return_indexes:
			return X, Y
		return X, Y, valid_indexes

	def train(self, ds_train, ds_test, nb_epoch=30):
		print('Prepraing text input...')
		text_train_data, text_train_y = self._prepare_text_input(ds_train)
		text_test_data, text_test_y = self._prepare_text_input(ds_test)
		print('X_train: ', text_train_data[0].shape, text_train_data[1].shape)
		print('Y_train: ', text_train_y.shape)
		print('X_test: ', text_test_data[0].shape, text_test_data[1].shape)
		print('Y_test: ', text_test_y.shape)

		print('Prepraing audio input...')
		audio_train_data, audio_train_y = self._prepare_audio_input(ds_train)
		audio_test_data, audio_test_y, indexes = self._prepare_audio_input(ds_test, return_indexes=True)
		print('X_train: ', audio_train_data[0].shape)
		print('Y_train: ', audio_train_y.shape)
		print('X_test: ', audio_test_data[0].shape)
		print('Y_test: ', audio_test_y.shape)

		classes = list(range(self.nb_classes))
		cw_text = dict(zip(classes, compute_class_weight('balanced', classes, unvectorize(text_train_y))))
		cw_audio = dict(zip(classes, compute_class_weight('balanced', classes, unvectorize(audio_train_y))))
		print('Class weights text: ', cw_text)
		print('Class weights audio: ', cw_audio)
		self._select_algorithm(cw_text=cw_text, cw_audio=cw_audio)

		logger.info('Training text %s...' % str(self.text_model))
		X = np.concatenate((text_train_data[0], text_train_data[1]), axis=-1)
		Xt = np.concatenate((text_test_data[0], text_test_data[1]), axis=-1)
		Y = unvectorize(text_train_y).reshape(X.shape[0], 1)
		self.text_model.fit(X, Y)

		predict = lambda x, y: unvectorize(x.predict_proba(y)) if self.which == 'svm' else x.predict(y) 
		pred_t = predict(self.text_model, Xt)
		Statistics.print_metrics(unvectorize(text_test_y).flatten(), pred_t)
		text_metrics = list(Statistics.get_metrics(unvectorize(text_test_y).flatten(), pred_t))
	
		logger.info('Training audio %s...' % str(self.audio_model))
		X = audio_train_data[0].reshape(audio_train_data[0].shape[0], audio_train_data[0].shape[1]*audio_train_data[0].shape[2])
		Y = unvectorize(audio_train_y).reshape(X.shape[0], 1)
		self.audio_model.fit(X, Y)

		Xta = audio_test_data[0].reshape(audio_test_data[0].shape[0], audio_test_data[0].shape[1]*audio_test_data[0].shape[2])
		pred_a = predict(self.audio_model, Xta)
		Statistics.print_metrics(unvectorize(audio_test_y).flatten(), pred_a)
		audio_metrics = list(Statistics.get_metrics(unvectorize(audio_test_y).flatten(), pred_a))

		text_test_data, _ = self._prepare_text_input(ds_test, valid_indexes=indexes)
		Xt = np.concatenate((text_test_data[0], text_test_data[1]), axis=-1)
		pred_a = self.audio_model.predict_proba(Xta)
		pred_t = self.text_model.predict_proba(Xt)

		logger.info('Evaluating...')
		gold = unvectorize(audio_test_y).flatten()
		partitions = list(map(lambda p: p/10.0, range(0, 11)))
		max_p = 0
		best_pred = []
		total_metrics = [0, 0, 0]

		for p in partitions:
			pred = unvectorize(p*pred_t + (1-p)*pred_a)
			logger.info('F1 for p: %.2f' % p)
			f1 = Statistics.print_metrics(gold, pred)
			if f1 > total_metrics[0]:
				total_metrics = list(Statistics.get_metrics(gold, pred))
				max_p = p
				best_pred = pred

		return text_metrics, audio_metrics, total_metrics, max_p, best_pred
