import numpy as np

import logging
from pprint import pformat

from keras.models import Model
from keras.layers import *

from deepbond.models.prosodic import ProsodicModel

logger = logging.getLogger(__name__)


class CNN(ProsodicModel):

	def _prepare_params(self):
		self.prosodic_size = self.features.prosodic.nb_phones * self.features.prosodic.nb_features

	def build(self, dropout_rate=0.5, nb_filter=100, filter_length=7, stride=1, pool_length=3, activation='relu', verbose=True):
		if verbose:
			logger.info('Building...')

		padding = filter_length // 2
		pool_padding = pool_length // 2
		inputs = []
		feats =	[]

		sequence_pros = Input(name='input_pros', shape=(self.input_length, self.prosodic_size))
		inputs.append(sequence_pros)
		
		cnn1d_pad 		= ZeroPadding1D(padding)(sequence_pros)
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