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


class BucketRNN:

	def __init__(self, vocabulary=None, features=None, nb_classes=2, batch_size=32, input_length=None):
		self.vocabulary = vocabulary
		self.features = features
		self.nb_classes = nb_classes
		self.batch_size = batch_size
		self.input_length = input_length
		self.model = None
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
