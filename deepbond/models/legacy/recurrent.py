import sys
import numpy as np
import logging

from pprint import pformat
from sklearn.utils import compute_class_weight

from keras import backend as K
from keras.models import Model
from keras.callbacks import Callback, RemoteMonitor
from keras.layers import *

from deepbond.utils import unroll, vectorize, unvectorize, row_matrix
from deepbond.statistics import Statistics

logger = logging.getLogger(__name__)


class RecurrentEnsemble:

	def __init__(self, vocabulary=None, features=None, nb_classes=2, nb_hidden=200, dropout_rate=0.5, batch_size=32, input_length=None, just_text=False, use_embeddings=True, use_pos=True):
		self.nb_classes = nb_classes
		self.nb_hidden = nb_hidden
		self.dropout_rate = dropout_rate
		self.input_length = input_length
		self.batch_size = batch_size
		self.vocabulary = vocabulary
		self.features = features
		self.model = None
		self.just_text = just_text
		self.use_embeddings = use_embeddings
		self.use_pos = use_pos
		self._prepare_params()

	def _prepare_params(self):
		self.vocabulary_size = max(self.vocabulary.values()) + 1
		self.embeddings_weights = np.array(self.features.get_embeddings(self.vocabulary))
		self.embeddings_size = self.embeddings_weights.shape[1]
		self.POS_vocab_size = max(self.features.pos.vocabulary.values()) + 10
		self.POS_embeddings_size = 10
		self.prosodic_size = self.features.prosodic.nb_phones * self.features.prosodic.nb_features

	def build(self):
		logger.info('Building model...')

		## text model
		sequence_text = Input(name='input_source', shape=(self.input_length,), dtype='int32')
		embedded_text = Embedding(self.vocabulary_size, self.embeddings_size, input_length=self.input_length, weights=[self.embeddings_weights])(sequence_text)
		
		sequence_pos = Input(name='input_pos', shape=(self.input_length,), dtype='int32')
		# embedded_pos = Reshape(input_length, 1)(sequence_pos)
		embedded_pos = Embedding(self.POS_vocab_size, self.POS_embeddings_size, input_length=self.input_length, init='glorot_normal')(sequence_pos)
		merge_embedded = merge([embedded_text, embedded_pos], mode='concat', concat_axis=-1)

		f_text = GRU(self.nb_hidden, return_sequences=True, activation='sigmoid')(merge_embedded)
		b_test = GRU(self.nb_hidden, return_sequences=True, go_backwards=True, activation='sigmoid')(merge_embedded)
		text_model = merge([f_text, b_test], mode='concat', concat_axis=-1)

		## audio model
		sequence_pros = Input(name='input_audio', shape=(self.input_length, self.prosodic_size))
		f_audio = GRU(self.nb_hidden, return_sequences=True, activation='sigmoid')(sequence_pros)
		b_audio = GRU(self.nb_hidden, return_sequences=True, go_backwards=True, activation='sigmoid')(sequence_pros)
		audio_model = merge([f_audio, b_audio], mode='concat', concat_axis=-1)

		merged_model = merge([text_model, audio_model], mode='concat', concat_axis=-1)
		# output = Activation('softmax')(merged_model)
		drop = Dropout(self.dropout_rate)(merged_model)
		output = TimeDistributed(Dense(output_dim=self.nb_classes, activation='softmax'), name='output_source')(drop)
		self.model = Model(input=[sequence_text, sequence_pos, sequence_pros], output=output)
		self._compile()

	def _compile(self):
		logger.info('Compiling...')
		self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')
		# self.model.compile(optimizer='rmsprop', loss=f1, metrics=['accuracy'])
		logger.debug('Model built: {}'.format(pformat(self.model.get_config())))

	def _prepare_input(self, ds, train=False, mask_prosody=True, statistics=True):
		X_text, Y = ds.as_matrix(ids=True)
		X_POS = self.features.get_POS(ds.indexes_to_words(X_text))
		X_mask_lines =  None
		if mask_prosody:
			X_mask_lines = list(map(len, X_text))
		X_pros = self.features.get_prosodic(ds.pros_texts, mask_lines=X_mask_lines)
		X_ext = [Y] if train else []
		X_data = [X_text, X_POS, X_pros] + X_ext
		if statistics:
			self.features.embeddings_statistics(ds.word_texts)
		return X_data, Y

	def _last_label_is_one(self, y):
		y[-1] = 1
		return y

	# def train_buckets(self, ds_train, ds_test, nb_epoch=30, verbose=True):
	
	def train(self, ds_train, ds_test, nb_epoch=30, verbose=True):
		return self.train_one_per_batch(ds_train, ds_test, nb_epoch=nb_epoch, verbose=verbose)

	def train_one_per_batch(self, ds_train, ds_test, nb_epoch=30, verbose=True):
		logger.info('Training...')
		
		train_data, train_y = self._prepare_input(ds_train, train=True)
		test_data, test_y = self._prepare_input(ds_test, train=False)
		
		Y_pred = []
		Y_gold = np.array(unroll(test_y))
		Y_gold_t = np.array(unroll(train_y))
		classes = [0, 1]
		classes_weight = dict(zip(classes, compute_class_weight('balanced', classes, Y_gold_t)))
		# logger.info('Class weights: {}'.format(class_weight))

		for e in range(nb_epoch):
			logger.debug('Epoch %d of %d:' % (e+1, nb_epoch))
			
			# train
			for i, data in enumerate(zip(*train_data)):
				sys.stdout.write('batch %d/%d \r' % (i+1, len(train_y)))
				sys.stdout.flush()
				X = list(map(row_matrix, data[:-1]))
				Y = vectorize(row_matrix(data[-1]), one_hot_dim=self.nb_classes)
				S = row_matrix(np.array(list(map(classes_weight.__getitem__, data[-1]))))
				self.model.train_on_batch(X, Y, sample_weight=S)
			
			# predict
			y = self.predict_on_batch(test_data)
			Y_pred = np.array(y).flatten()
			Y_pred_last = self._last_label_is_one(np.array(y)).flatten()
			Y_pred_t = np.array(self.predict_on_batch(train_data[:-1])).flatten()
			
			# statistics
			if verbose:
				Statistics.print_metrics(Y_gold, Y_pred)
				Statistics.print_metrics(Y_gold, Y_pred_last)
				Statistics.print_metrics(Y_gold_t, Y_pred_t)

	def predict_on_batch(self, test_data):
		Y_pred = []
		for data in zip(*test_data):
			X = list(map(row_matrix, data))
			pred = unvectorize(self.model.predict_on_batch(X))
			Y_pred.extend(pred[0])
		return Y_pred

	def evaluate(self, ds_test):
		test_data, test_y = self._prepare_input(ds_test, train=False)
		logger.info('Evaluating...')
		Y_gold = np.array([y for x in test_y for y in x])
		Y_pred = np.array(self.predict_on_batch(test_data)).flatten()
		f1_1 = Statistics.print_metrics(Y_gold, Y_pred, print_cm=True)
		return f1_1

	def save(self):
		json_string = self.model.to_json()
		name = type(self).__name__
		open('data/architecture/'+name+'.json', 'w').write(json_string)
		self.model.save_weights('data/architecture/'+name+'_weights.h5')

	def load(self):
		from keras.models import model_from_json
		name = type(self).__name__
		m = 'data/architecture/'+name+'.json'
		w = 'data/architecture/'+name+'_weights.h5'
		self.model = model_from_json(open(m).read())
		self.model.load_weights(w)
		self._compile()



class RecurrentEnsembleDivided(RecurrentEnsemble):
	
	def build(self):
		logger.info('Building...')

		## text model
		sequence_text = Input(name='input_source', shape=(self.input_length,), dtype='int32')
		embedded_text = Embedding(self.vocabulary_size, self.embeddings_size, input_length=self.input_length, weights=[self.embeddings_weights])(sequence_text)
		
		sequence_pos = Input(name='input_pos', shape=(self.input_length,), dtype='int32')
		embedded_pos = Embedding(self.POS_vocab_size, self.POS_embeddings_size, input_length=self.input_length, init='glorot_normal')(sequence_pos)
		merge_embedded = merge([embedded_text, embedded_pos], mode='concat', concat_axis=-1)
		
		f_text = GRU(self.nb_hidden, return_sequences=True, activation='sigmoid')(merge_embedded)
		b_test = GRU(self.nb_hidden, return_sequences=True, go_backwards=True, activation='sigmoid')(merge_embedded)
		text_model = merge([f_text, b_test], mode='sum', concat_axis=-1)
		# text_output = Activation('softmax')(text_model)
		text_drop = Dropout(self.dropout_rate)(text_model)
		text_output = TimeDistributed(Dense(output_dim=self.nb_classes, activation='softmax'))(text_drop)
		self.text_model = Model(input=[sequence_text, sequence_pos], output=text_output)

		## audio model
		self.prosodic_size = self.features.prosodic.nb_phones * self.features.prosodic.nb_features
		sequence_pros = Input(name='input_audio', shape=(self.input_length, self.prosodic_size))
		
		f_audio = GRU(self.nb_hidden, return_sequences=True, activation='sigmoid')(sequence_pros)
		b_audio = GRU(self.nb_hidden, return_sequences=True, go_backwards=True, activation='sigmoid')(sequence_pros)
		audio_model = merge([f_audio, b_audio], mode='sum', concat_axis=-1)
		# audio_output = Activation('softmax')(audio_model)
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

	def train_one_per_batch(self, ds_train, ds_test, nb_epoch=30, verbose=True):
		logger.info('Training...')
		
		train_data, train_y = self._prepare_input(ds_train, train=True, mask_prosody=False)
		test_data, test_y = self._prepare_input(ds_test, train=False, mask_prosody=False)
		
		Y_pred = []
		Y_gold = np.array(unroll(test_y))
		Y_gold_t = np.array(unroll(train_y))
		classes = [0, 1]
		classes_weight = dict(zip(classes, compute_class_weight('balanced', classes, Y_gold_t)))
		# print('Class weights: {}'.format(classes_weight))

		print('X_train: ', len(train_data[0]), len(train_data[1]), len(train_data[2]))
		print('Y_train: ', len(train_y))
		print('X_test: ', len(test_data[0]), len(test_data[1]), len(test_data[2]))
		print('Y_test: ', len(test_y))
		max_all, max_text, max_audio, max_p = [0,0,0], [0,0,0], [0,0,0], 0

		for e in range(nb_epoch):
			logger.debug('Epoch %d of %d:' % (e+1, nb_epoch))
			
			# train
			for i, data in enumerate(zip(*train_data)):
				sys.stdout.write('batch %d/%d \r' % (i+1, len(train_y)))
				sys.stdout.flush()
				X = list(map(row_matrix, data[:-1]))
				Y = vectorize(row_matrix(data[-1]), one_hot_dim=self.nb_classes)
				X_text = [X[0], X[1]]
				X_audio = [X[2]]
				S = row_matrix(np.array(list(map(classes_weight.__getitem__, data[-1]))))
				if data[2] is not None:
					self.audio_model.train_on_batch(X_audio, Y, sample_weight=S)
				self.text_model.train_on_batch(X_text, Y, sample_weight=S)
			
			# predict
			y_text = self.predict_on_batch(test_data, what='text')
			if not self.just_text:
				y_all, max_partition = self.evaluate_on_batch(test_data, Y_gold)
				y_audio = self.predict_on_batch(test_data, what='audio')

				Y_pred_all = np.array(y_all).flatten()
				Y_pred_last_all = self._last_label_is_one(np.array(y_all)).flatten()

				Y_pred_audio = np.array(y_audio).flatten()
				Y_pred_last_audio = self._last_label_is_one(np.array(y_audio)).flatten()

			Y_pred_text = np.array(y_text).flatten()
			Y_pred_last_text = self._last_label_is_one(np.array(y_text)).flatten()
			
			Y_pred_t = np.array(self.predict_on_batch(train_data[:-1])).flatten()
			
			# statistics
			if verbose:
				if not self.just_text:
					logger.info('Max F1 for p: %.2f' % max_partition)
					# Statistics.print_metrics(Y_gold, Y_pred_all)
					metrics_all = list(Statistics.get_metrics(Y_gold, Y_pred_last_all))
					last_max_all_f1 = max_all[0]
					if metrics_all[0] > max_all[0]:
						max_all = metrics_all
						Y_pred = Y_pred_last_all
					if last_max_all_f1 != max_all[0]:
						max_p = max_partition
					Statistics.print_metrics(Y_gold, Y_pred_last_all) 
				
				# Statistics.print_metrics(Y_gold, Y_pred_text)
				metrics_text = list(Statistics.get_metrics(Y_gold, Y_pred_last_text))
				if metrics_text[0] > max_text[0]:
					max_text = metrics_text
				Statistics.print_metrics(Y_gold, Y_pred_last_text)

				if not self.just_text:
					# Statistics.print_metrics(Y_gold, Y_pred_audio)
					metrics_audio = list(Statistics.get_metrics(Y_gold, Y_pred_last_audio))
					if metrics_audio[0] > max_audio[0]:
						max_audio = metrics_audio
					Statistics.print_metrics(Y_gold, Y_pred_last_audio)

				Statistics.print_metrics(Y_gold_t, Y_pred_t)

		logger.debug('Max F1 for p: %.1f' % max_p)
		logger.debug("All Precision = %.3f  Recall = %.3f F-Measure = %.3f" % (max_all[1], max_all[2], max_all[0]))
		logger.debug("Text Precision = %.3f  Recall = %.3f F-Measure = %.3f" % (max_text[1], max_text[2], max_text[0]))
		logger.debug("Audio Precision = %.3f  Recall = %.3f F-Measure = %.3f" % (max_audio[1], max_audio[2], max_audio[0]))
		
		return max_text, max_audio, max_all, max_p, Y_pred

	def predict_on_batch(self, test_data, what='all'):
		Y_pred = []
		for data in zip(*test_data):
			X = list(map(row_matrix, data))
			X_text = [X[0], X[1]]
			X_audio = [X[2]]
			has_audio = data[2] is not None
			if what == 'all' and has_audio:
				pred_text = self.text_model.predict_on_batch(X_text)
				pred_audio = self.audio_model.predict_on_batch(X_audio)
				pred_all = 0.7*pred_text + 0.3*pred_audio
				pred = unvectorize(pred_all)
			elif what == 'text':
				pred = unvectorize(self.text_model.predict_on_batch(X_text))
			elif what == 'audio':
				pred = unvectorize(self.audio_model.predict_on_batch(X_audio))
			else:
				pred = unvectorize(self.text_model.predict_on_batch(X_text))
			Y_pred.extend(pred[0])
			Y_pred[-1] = 1
		return Y_pred

	def evaluate_on_batch(self, test_data, Y_gold):
		partitions = list(map(lambda p: p/10.0, range(0, 11)))
		Y_pred = [[] for _ in range(len(partitions))]
		for i, x in enumerate(partitions):
			for data in zip(*test_data):
				X = list(map(row_matrix, data))
				X_text = [X[0], X[1]]
				X_audio = [X[2]]
				pred_text = self.text_model.predict_on_batch(X_text)
				pred_audio = self.audio_model.predict_on_batch(X_audio)
				pred_all = x*pred_text + (1-x)*pred_audio
				pred = unvectorize(pred_all)
				Y_pred[i].extend(pred[0])
				Y_pred[i][-1] = 1
		max_p_index, max_f_value = 0, 0
		for i in range(len(partitions)):
			f1, prec, rec = Statistics.get_metrics(Y_gold, Y_pred[i])
			if f1 > max_f_value:
				max_p_index = i
				max_f_value = f1
		return Y_pred[max_p_index], partitions[max_p_index]

	def evaluate(self, ds_test):
		test_data, test_y = self._prepare_input(ds_test, train=False)
		partitions = list(map(lambda p: p/10.0, range(0, 11)))
		Y_pred = [[] for _ in range(len(partitions))]
		Y_gold = np.array(unroll(test_y))
		logger.info('Evaluating...')
		for data in zip(*test_data):
			X = list(map(row_matrix, data))
			X_text = [X[0], X[1]]
			X_audio = [X[2]]
			for i, x in enumerate(partitions):
				pred_text = self.text_model.predict_on_batch(X_text)
				pred_audio = self.audio_model.predict_on_batch(X_audio)
				pred_all = x*pred_text + (1-x)*pred_audio
				pred = unvectorize(pred_all)
				Y_pred[i].extend(pred[0])
				Y_pred[i][-1] = 1
		for i in range(len(partitions)):
			logger.info('F1 for p: %.2f' % partitions[i])
			Statistics.print_metrics(Y_gold, Y_pred[i])


