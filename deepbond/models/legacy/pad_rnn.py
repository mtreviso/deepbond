import numpy as np
import logging

from pprint import pformat
from sklearn.utils import compute_class_weight

from keras.models import Model
from keras.layers import *

from deepbond.utils import unroll, vectorize, unvectorize, pad_sequences, pad_sequences_3d, unpad_sequences
from deepbond.statistics import Statistics
from deepbond.models.callbacks import TrainStats

logger = logging.getLogger(__name__)


class RecurrentPadded:

	def __init__(self, vocabulary=None, features=None, nb_classes=2, nb_hidden=200, dropout_rate=0.5, batch_size=32, input_length=None, use_embeddings=True, use_pos=True):
		self.nb_classes = nb_classes
		self.nb_hidden = nb_hidden
		self.dropout_rate = dropout_rate
		self.input_length = input_length
		self.batch_size = batch_size
		self.vocabulary = vocabulary
		self.features = features
		self.model = None
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
		embedded_text = Embedding(self.vocabulary_size, self.embeddings_size, input_length=self.input_length, weights=[self.embeddings_weights])(sequence_text)
		
		sequence_pos = Input(name='input_pos', shape=(self.input_length,), dtype='int32')
		embedded_pos = Embedding(self.POS_vocab_size, self.POS_embeddings_size, input_length=self.input_length, init='he_normal')(sequence_pos)
		merge_embedded = merge([embedded_text, embedded_pos], mode='concat', concat_axis=-1)
		
		f_text = LSTM(self.nb_hidden, return_sequences=True, activation='sigmoid')(merge_embedded)
		b_test = LSTM(self.nb_hidden, return_sequences=True, go_backwards=True, activation='sigmoid')(merge_embedded)
		text_model = merge([f_text, b_test], mode='sum', concat_axis=-1)
		text_drop = Dropout(self.dropout_rate)(text_model)
		text_output = TimeDistributed(Dense(output_dim=self.nb_classes, activation='softmax'))(text_drop)
		self.text_model = Model(input=[sequence_text, sequence_pos], output=text_output)

		# audio model
		self.prosodic_size = self.features.prosodic.nb_phones * self.features.prosodic.nb_features
		sequence_pros = Input(name='input_audio', shape=(self.input_length, self.prosodic_size))
		
		f_audio = LSTM(self.nb_hidden, return_sequences=True, activation='sigmoid')(sequence_pros)
		b_audio = LSTM(self.nb_hidden, return_sequences=True, go_backwards=True, activation='sigmoid')(sequence_pros)
		audio_model = merge([f_audio, b_audio], mode='sum', concat_axis=-1)
		audio_drop = Dropout(self.dropout_rate)(audio_model)
		audio_output = TimeDistributed(Dense(output_dim=self.nb_classes, activation='softmax'))(audio_drop)
		self.audio_model = Model(input=[sequence_pros], output=audio_output)

		self._compile()

	def _compile(self):
		logger.info('Compiling...')
		self.text_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')
		self.audio_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')
		logger.debug('Model text built: {}'.format(pformat(self.text_model.get_config())))
		logger.debug('Model audio built: {}'.format(pformat(self.audio_model.get_config())))

	def _prepare_text_input(self, ds, valid_indexes=None):
		X_text, Y_gold = ds.as_matrix(ids=True)
		if valid_indexes is not None:
			X_text = [X_text[i] for i in valid_indexes]
			Y_gold = [Y_gold[i] for i in valid_indexes]
		X_POS = self.features.get_POS(ds.indexes_to_words(X_text))
		X_text = pad_sequences(X_text, maxlen=self.input_length, mask_value=0)
		X_POS = pad_sequences(X_POS, maxlen=self.input_length, mask_value=0)
		Y = vectorize(pad_sequences(Y_gold, maxlen=self.input_length, mask_value=0), one_hot_dim=self.nb_classes)
		return [X_text, X_POS], Y, Y_gold

	def _prepare_audio_input(self, ds, return_indexes=False):
		_, Y_gold = ds.as_matrix(ids=True)
		X_audio = self.features.get_prosodic(ds.pros_texts, mask_lines=None)
		valid_indexes = [i for i, x in enumerate(X_audio) if x != None]
		X_audio = [X_audio[i] for i in valid_indexes]
		Y_gold = [Y_gold[i] for i in valid_indexes]
		X_audio = pad_sequences_3d(X_audio, maxlen=self.input_length, mask_value=0)
		Y = vectorize(pad_sequences(Y_gold, maxlen=self.input_length, mask_value=0), one_hot_dim=self.nb_classes)
		if return_indexes:
			return [X_audio], Y, Y_gold, valid_indexes
		return [X_audio], Y, Y_gold

	def train(self, ds_train, ds_test, nb_epoch=20):
		logger.info('Training...')
		self.features.embeddings_statistics(ds_train.word_texts)
		self.features.embeddings_statistics(ds_test.word_texts)

		print('Prepraing text input...')
		text_train_data, text_train_y, text_train_gold = self._prepare_text_input(ds_train)
		text_test_data, text_test_y, text_test_gold = self._prepare_text_input(ds_test)
		print('X_train: ', text_train_data[0].shape, text_train_data[1].shape)
		print('Y_train: ', text_train_y.shape)
		print('X_test: ', text_test_data[0].shape, text_test_data[1].shape)
		print('Y_test: ', text_test_y.shape)
		
		print('Prepraing audio input...')
		audio_train_data, audio_train_y, audio_train_gold = self._prepare_audio_input(ds_train)
		audio_test_data, audio_test_y, audio_test_gold = self._prepare_audio_input(ds_test)
		print('X_train: ', audio_train_data[0].shape)
		print('Y_train: ', audio_train_y.shape)
		print('X_test: ', audio_test_data[0].shape)
		print('Y_test: ', audio_test_y.shape)

		classes = list(range(text_train_y.shape[-1]))
		classes_weight = dict(zip(classes, compute_class_weight('balanced', classes, np.array(unroll(unvectorize(text_train_y))))))
		sample_weight_text = np.array(list(map(lambda x: list(map(classes_weight.__getitem__, x)), unvectorize(text_train_y))))
		sample_weight_audio = np.array(list(map(lambda x: list(map(classes_weight.__getitem__, x)), unvectorize(audio_train_y))))
		
		text_callbacks = [TrainStats(text_train_data, text_train_y, text_test_data, text_test_y, self.batch_size, model=self.text_model, parent=self, f_unpad=unpad_sequences, f_data=(text_train_gold, text_test_gold))]
		self.text_model.fit(text_train_data, text_train_y, nb_epoch=nb_epoch, batch_size=self.batch_size, callbacks=text_callbacks, sample_weight=sample_weight_text)

		audio_callbacks = [TrainStats(audio_train_data, audio_train_y, audio_test_data, audio_test_y, self.batch_size, model=self.audio_model, parent=self, eval_parent=True, f_unpad=unpad_sequences, f_data=(audio_train_gold, audio_test_gold))]
		self.audio_model.fit(audio_train_data, audio_train_y, nb_epoch=nb_epoch, batch_size=self.batch_size, callbacks=audio_callbacks, sample_weight=sample_weight_audio)

		total_metrics, self.max_p, best_all = self.evaluate(text_test_gold)
		text_metrics  = list(Statistics.get_metrics(unroll(text_test_gold), unvectorize(unpad_sequences(self.best_text, text_test_gold)).flatten()))
		audio_metrics = list(Statistics.get_metrics(unroll(audio_test_gold), unvectorize(unpad_sequences(self.best_audio, audio_test_gold)).flatten()))
		return text_metrics, audio_metrics, total_metrics, self.max_p, best_all
		
	def predict(self, ds_test):
		audio_test_data, _, audio_gold, indexes = self._prepare_audio_input(ds_test, return_indexes=True)
		text_test_data, _, text_gold			= self._prepare_text_input(ds_test, valid_indexes=indexes)
		pred_text = self.text_model.predict(text_test_data, batch_size=self.batch_size)
		pred_audio = self.text_model.predict(audio_test_data, batch_size=self.batch_size)
		return unvectorize(self.max_p*unpad_sequences(pred_text, text_gold) + (1-self.max_p)*unpad_sequences(pred_audio, audio_gold))

	def evaluate(self, gold):
		partitions = list(map(lambda p: p/10.0, range(0, 11)))
		pred_text = self.best_text
		pred_audio = self.best_audio
		Y_gold = unroll(gold)
		best_pred = None
		total_metrics = [0,0,0]
		max_p = 1
		for p in partitions:
			Y_pred = unvectorize(p*unpad_sequences(pred_text, gold) + (1-p)*unpad_sequences(pred_audio, gold)).flatten()
			logger.info('F1 for p: %.2f' % p)
			f1 = Statistics.print_metrics(Y_gold, Y_pred)
			if f1 > total_metrics[0]:
				total_metrics = list(Statistics.get_metrics(Y_gold, Y_pred))
				max_p = p
				best_pred = Y_pred
		return total_metrics, max_p, best_pred

	def save(self):
		pass

	def load(self):
		pass

