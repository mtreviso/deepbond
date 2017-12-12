import numpy as np
import logging

from keras import backend as K
from keras.models import Model
from keras.layers import *

from deepbond.statistics import Statistics
from deepbond.models import BucketRNN

logger = logging.getLogger(__name__)


class BucketRCNN(BucketRNN):

	def build(self, filter_length=7, rnn='GRU', max_pooling=True):
		logger.info('Building...')

		filter_length = filter_length
		padding = filter_length // 2
		nb_filter_text = 100
		nb_filter_audio = 4
		stride = 1
		pool_length = 3
		pool_padding = pool_length // 2
		border_mode = 'valid'
		RNN = LSTM if rnn == 'LSTM' else GRU
		self.max_pooling = max_pooling

		## text model
		sequence_text = Input(name='input_source', shape=(self.input_length,), dtype='int32')
		sequence_pos = Input(name='input_pos', shape=(self.input_length,), dtype='int32')
		if self.use_embeddings:
			embedded_text = Embedding(self.vocabulary_size, self.embeddings_size, input_length=self.input_length, weights=[self.embeddings_weights])(sequence_text)
		else:
			embedded_text = Embedding(self.vocabulary_size, self.embeddings_size, input_length=self.input_length, init='uniform', trainable=False)(sequence_text)
		if self.use_pos:
			embedded_pos = Embedding(self.POS_vocab_size, self.POS_embeddings_size, input_length=self.input_length, init='glorot_normal')(sequence_pos)
		else:
			embedded_pos = Embedding(self.POS_vocab_size, 1, input_length=self.input_length, init='zero', trainable=False)(sequence_pos)
		merge_embedded = merge([embedded_text, embedded_pos], mode='concat', concat_axis=-1)
		cnn1d_text_pad = ZeroPadding1D(padding)(merge_embedded)
		cnn1d_text = Convolution1D(nb_filter=nb_filter_text, filter_length=filter_length, 
								border_mode=border_mode, subsample_length=stride, activation='relu')(cnn1d_text_pad)
		if max_pooling:
			# maxpooling_text = Lambda(max_1d, output_shape=(self.input_length, 1))(cnn1d_text)	
			maxpooling_text_pad = ZeroPadding1D(pool_padding)(cnn1d_text)
			maxpooling_text = MaxPooling1D(pool_length=pool_length, border_mode=border_mode, stride=stride)(maxpooling_text_pad)
		else:
			maxpooling_text = cnn1d_text
		f_text = RNN(self.nb_hidden, return_sequences=True, activation='sigmoid')(maxpooling_text)	
		b_text = RNN(self.nb_hidden, return_sequences=True, go_backwards=True, activation='sigmoid')(maxpooling_text)
		text_model = merge([f_text, b_text], mode='sum', concat_axis=-1)
		text_drop = Dropout(self.dropout_rate)(text_model)
		text_output = TimeDistributed(Dense(output_dim=self.nb_classes, activation='softmax'))(text_drop)
		self.text_model = Model(input=[sequence_text, sequence_pos], output=text_output)


		
		## audio model
		self.prosodic_size = self.features.prosodic.nb_phones * self.features.prosodic.nb_features
		sequence_pros = Input(name='input_audio', shape=(self.input_length, self.prosodic_size))
		cnn1d_audio_pad = ZeroPadding1D(padding)(sequence_pros)
		cnn1d_audio = Convolution1D(nb_filter=nb_filter_audio, filter_length=filter_length, border_mode=border_mode, 
									subsample_length=stride, activation='relu')(cnn1d_audio_pad)
		if max_pooling:
			# maxpooling_audio = Lambda(max_1d, output_shape=(self.input_length, 1))(cnn1d_audio)
			maxpooling_audio_pad = ZeroPadding1D(pool_padding)(cnn1d_audio)
			maxpooling_audio = MaxPooling1D(pool_length=pool_length, border_mode=border_mode, stride=stride)(maxpooling_audio_pad)
		else:
			maxpooling_audio = cnn1d_audio
		f_audio = RNN(self.nb_hidden, return_sequences=True, activation='sigmoid')(maxpooling_audio)
		b_audio = RNN(self.nb_hidden, return_sequences=True, go_backwards=True, activation='sigmoid')(maxpooling_audio)
		audio_model = merge([f_audio, b_audio], mode='sum', concat_axis=-1)
		audio_drop = Dropout(self.dropout_rate)(audio_model)
		audio_output = TimeDistributed(Dense(output_dim=self.nb_classes, activation='softmax'))(audio_drop)
		self.audio_model = Model(input=[sequence_pros], output=audio_output)


		self._compile()
		self.text_model.summary()

