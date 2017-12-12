import numpy as np

import logging
from pprint import pformat

from keras.models import Model
from keras.layers import *

from deepbond.models.lexical import LexicalModel

logger = logging.getLogger(__name__)


class CNN(LexicalModel):

	def _prepare_params(self):
		if self.use_embeddings:
			self.emb_weights = np.array(self.features.get_embeddings(self.vocabulary))
			self.emb_vocab_size = self.emb_weights.shape[0]
			self.emb_size = self.emb_weights.shape[1]
		
		if self.use_pos:
			self.POS_vocab_size = max(self.features.pos.vocabulary.values()) + 1
			self.POS_emb_size = 10

		if self.use_handcrafted:
			self.hc_size = self.features.handcrafted.dimensions


	def build(self, dropout_rate=0.5, nb_filter=100, filter_length=7, stride=1, pool_length=3, activation='relu', verbose=True):
		if verbose:
			logger.info('Building...')

		padding = filter_length // 2
		pool_padding = pool_length // 2
		inputs = []
		feats =	[]

		if self.use_embeddings:
			sequence = Input(name='input_emb', shape=(self.input_length,), dtype='int32')
			embedded = Embedding(self.emb_vocab_size, self.emb_size, input_length=self.input_length, weights=[self.emb_weights])(sequence)
			inputs.append(sequence)
			feats.append(embedded)

		if self.use_pos:
			sequence_pos = Input(name='input_pos', shape=(self.input_length,), dtype='int32')
			embedded_pos = Embedding(self.POS_vocab_size, self.POS_emb_size, input_length=self.input_length)(sequence_pos)
			inputs.append(sequence_pos)
			feats.append(embedded_pos)
		
		if self.use_handcrafted:
			sequence_hc = Input(name='input_hc', shape=(self.input_length, self.hc_size))
			inputs.append(sequence_hc)
			feats.append(sequence_hc)
		
		if sum([self.use_embeddings, self.use_pos, self.use_handcrafted]) > 1:
			merge_features 	= merge(feats, mode='concat', concat_axis=-1)
		elif self.use_embeddings:
			merge_features = embedded
		elif self.use_pos:
			merge_features = embedded_pos
		
		cnn1d_pad 		= ZeroPadding1D(padding)(merge_features)
		cnn1d 			= Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation=activation, 
										subsample_length=stride, border_mode='valid')(cnn1d_pad)
	
		maxpooling_pad 	= ZeroPadding1D(pool_padding)(cnn1d)
		maxpooling 		= MaxPooling1D(pool_length=pool_length, border_mode='valid', stride=stride)(maxpooling_pad)

		drop 			= Dropout(dropout_rate)(maxpooling)
		output 			= TimeDistributed(Dense(output_dim=self.nb_classes, activation='softmax'), name='output_source')(drop)
		self.classifier = Model(input=inputs, output=output)

		self._compile()
		if verbose:
			self._summary()
	
	def _compile(self):
		logger.info('Compiling...')
		self.classifier.compile(optimizer='adam', loss='categorical_crossentropy', sample_weight_mode='temporal')

	def _summary(self):
		self.classifier.summary()
		logger.debug('Model built: {}'.format(pformat(self.classifier.get_config())))