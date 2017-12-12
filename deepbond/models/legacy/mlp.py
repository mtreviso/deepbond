import numpy as np
import logging

from pprint import pformat
from sklearn.utils import compute_class_weight

from keras import backend as K
from keras.models import Model
from keras.callbacks import Callback
from keras.layers import *

from deepbond.utils import unroll, vectorize, unvectorize, column_matrix, convolve_sequences, convolve_sequences_3d
from deepbond.statistics import Statistics
from deepbond.models.callbacks import TrainStats

logger = logging.getLogger(__name__)


class WindowEnsembleMLP:
	
	def __init__(self, vocabulary=None, features=None, nb_classes=2, nb_hidden=200, dropout_rate=0.5, input_length=5, batch_size=32, use_embeddings=True, use_pos=True):
		self.nb_classes = nb_classes
		self.nb_hidden = nb_hidden
		self.dropout_rate = dropout_rate
		self.input_length = input_length
		self.vocabulary = vocabulary
		self.features = features
		self.batch_size = batch_size
		self.use_embeddings = use_embeddings
		self.use_pos = use_pos
		self.best_text = None
		self.best_audio = None
		self.max_p = 1
		self._prepare_params()

	def _prepare_params(self):
		self.vocabulary_size = max(self.vocabulary.values()) + 1
		self.embeddings_weights = np.array(self.features.get_embeddings(self.vocabulary))
		self.embeddings_size = self.embeddings_weights.shape[1]
		self.POS_vocab_size = max(self.features.pos.vocabulary.values()) + 10
		self.POS_embeddings_size = 10
		self.prosodic_size = self.features.prosodic.nb_phones * self.features.prosodic.nb_features

	def build(self):
		logger.info('Building...')

		# text model
		sequence_text = Input(name='input_source', shape=(self.input_length,), dtype='int32')
		if self.use_embeddings:
			embedded_text = Embedding(self.vocabulary_size, self.embeddings_size, input_length=self.input_length, weights=[self.embeddings_weights])(sequence_text)
		else:
			embedded_text = Embedding(self.vocabulary_size, self.embeddings_size, input_length=self.input_length, init='uniform', trainable=False)(sequence_text)
		sequence_pos = Input(name='input_pos', shape=(self.input_length,), dtype='int32')
		if self.use_pos:
			embedded_pos = Embedding(self.POS_vocab_size, self.POS_embeddings_size, input_length=self.input_length, init='glorot_normal')(sequence_pos)
		else:
			embedded_pos = Embedding(self.POS_vocab_size, 1, input_length=self.input_length, init='zero', trainable=False)(sequence_pos)
		merge_embedded = merge([embedded_text, embedded_pos], mode='concat', concat_axis=-1)
		text_flattened = Flatten()(merge_embedded)
		text_dense = Dense(self.nb_hidden, activation='sigmoid')(text_flattened)
		text_drop = Dropout(self.dropout_rate)(text_dense)
		text_dense2 = Dense(self.nb_hidden // 2, activation='sigmoid')(text_drop)
		text_drop2 = Dropout(self.dropout_rate)(text_dense2)
		text_output = Dense(self.nb_classes, activation='softmax')(text_drop2)
		self.text_model = Model(input=[sequence_text, sequence_pos], output=text_output)

		# audio model
		self.prosodic_size = self.features.prosodic.nb_phones * self.features.prosodic.nb_features
		sequence_audio = Input(name='input_audio', shape=(self.input_length, self.prosodic_size))
		audio_flattened = Flatten()(sequence_audio)
		audio_dense = Dense(self.nb_hidden, activation='sigmoid')(audio_flattened)
		audio_drop = Dropout(self.dropout_rate)(audio_dense)
		audio_dense2 = Dense(self.nb_hidden // 2, activation='sigmoid')(audio_drop)
		audio_drop2 = Dropout(self.dropout_rate)(audio_dense2)
		audio_output = Dense(self.nb_classes, activation='softmax')(audio_drop2)
		self.audio_model = Model(input=[sequence_audio], output=audio_output)

		self._compile()

	def _compile(self):
		logger.info('Compiling...')
		self.text_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
		self.audio_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
		logger.debug('Model text built: {}'.format(pformat(self.text_model.get_config())))
		logger.debug('Model audio built: {}'.format(pformat(self.audio_model.get_config())))

	def _prepare_text_input(self, ds, valid_indexes=None):
		X_text, Y = ds.as_matrix(ids=True)
		if valid_indexes is not None:
			X_text = [X_text[i] for i in valid_indexes]
			Y = [Y[i] for i in valid_indexes]
		X_POS = self.features.get_POS(ds.indexes_to_words(X_text))
		X_text = convolve_sequences(X_text, self.input_length, left_pad_value=0, right_pad_value=0)
		X_POS = convolve_sequences(X_POS, self.input_length, left_pad_value=0, right_pad_value=0)
		X = [X_text, X_POS]
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

	def train(self, ds_train, ds_test, nb_epoch=50):
		logger.info('Training...')
		self.features.embeddings_statistics(ds_train.word_texts)
		self.features.embeddings_statistics(ds_test.word_texts)

		print('Prepraing text input...')
		text_train_data, text_train_y = self._prepare_text_input(ds_train)
		text_test_data, text_test_y = self._prepare_text_input(ds_test)
		print('X_train: ', text_train_data[0].shape, text_train_data[1].shape)
		print('Y_train: ', text_train_y.shape)
		print('X_test: ', text_test_data[0].shape, text_test_data[1].shape)
		print('Y_test: ', text_test_y.shape)
		print('Chance baseline: ', Statistics.chance_baseline(np.hstack((unvectorize(text_train_y), unvectorize(text_test_y)))))
		
		
		print('Prepraing audio input...')
		audio_train_data, audio_train_y = self._prepare_audio_input(ds_train)
		audio_test_data, audio_test_y = self._prepare_audio_input(ds_test)
		print('X_train: ', audio_train_data[0].shape)
		print('Y_train: ', audio_train_y.shape)
		print('X_test: ', audio_test_data[0].shape)
		print('Y_test: ', audio_test_y.shape)
		print('Chance baseline: ', Statistics.chance_baseline(np.hstack((unvectorize(audio_train_y), unvectorize(audio_test_y)))))
		
		# add class_weight
		classes = list(range(self.nb_classes))
		C_t = dict(zip(classes, compute_class_weight('balanced', classes, unvectorize(text_train_y))))
		C_a = dict(zip(classes, compute_class_weight('balanced', classes, unvectorize(audio_train_y))))
		print(C_t, C_a)

		logger.info('Training text...')
		text_callbacks = [TrainStats(text_train_data, text_train_y, text_test_data, text_test_y, self.batch_size, model=self.text_model, parent=self)]
		self.text_model.fit(text_train_data, text_train_y, nb_epoch=nb_epoch, batch_size=self.batch_size, callbacks=text_callbacks, class_weight=C_t)

		logger.info('Training audio...')
		audio_callbacks = [TrainStats(audio_train_data, audio_train_y, audio_test_data, audio_test_y, self.batch_size, model=self.audio_model, parent=self, eval_parent=True)]
		self.audio_model.fit(audio_train_data, audio_train_y, nb_epoch=nb_epoch, batch_size=self.batch_size, callbacks=audio_callbacks, class_weight=C_a)

		logger.info('Evaluating...')
		total_metrics, self.max_p, best_all = self.evaluate(ds_test, batch_size=self.batch_size)

		pred_text = self.text_model.predict(text_test_data, batch_size=self.batch_size)
		text_metrics = list(Statistics.get_metrics(unvectorize(text_test_y), unvectorize(pred_text)))
		
		pred_audio = self.audio_model.predict(audio_test_data, batch_size=self.batch_size)
		audio_metrics = list(Statistics.get_metrics(unvectorize(audio_test_y), unvectorize(pred_audio)))

		return text_metrics, audio_metrics, total_metrics, self.max_p, best_all

	def evaluate(self, ds_test, batch_size=64):
		audio_test_data, audio_test_y, indexes = self._prepare_audio_input(ds_test, return_indexes=True)
		text_test_data, _ = self._prepare_text_input(ds_test, valid_indexes=indexes)
		partitions = list(map(lambda p: p/10.0, range(0, 11)))
		Y_gold = unvectorize(audio_test_y).flatten()
		pred_text = self.text_model.predict(text_test_data, batch_size=batch_size)
		pred_audio = self.audio_model.predict(audio_test_data, batch_size=batch_size)
		Y_pred = []
		total_metrics = [0, 0, 0]
		max_p = 1
		best_all = []
		for p in partitions:
			pred_all = p*pred_text + (1-p)*pred_audio
			Y_pred = unvectorize(pred_all)
			logger.info('F1 for p: %.2f' % p)
			f1 = Statistics.print_metrics(Y_gold, Y_pred)
			if f1 > total_metrics[0]:
				total_metrics = list(Statistics.get_metrics(Y_gold, Y_pred))
				max_p = p
				best_all = Y_pred
		return total_metrics, max_p, best_all
