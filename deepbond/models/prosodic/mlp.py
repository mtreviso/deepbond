import numpy as np

import logging
from pprint import pformat

from keras.models import Model
from keras.layers import *

from deepbond.models.prosodic import ProsodicModel

logger = logging.getLogger(__name__)


class MLP(ProsodicModel):

	def _prepare_params(self):
		self.prosodic_size = self.features.prosodic.nb_phones * self.features.prosodic.nb_features

	def build(self, nb_hidden=100, mlp_activation='sigmoid', dropout_rate=0.5, verbose=True):
		if verbose:
			logger.info('Building...')

		inputs = []
		feats =	[]

		sequence_pros = Input(name='input_pros', shape=(self.input_length, self.prosodic_size))
		inputs.append(sequence_pros)

		flattened 		= Flatten()(sequence_pros)
		hidden 			= Dense(output_dim=nb_hidden, activation=mlp_activation)(flattened)
		drop 			= Dropout(dropout_rate)(hidden)
		output 			= Dense(output_dim=self.nb_classes, activation='softmax', name='output_source')(drop)
		self.classifier = Model(input=inputs, output=output)

		if verbose:
			logger.info('Compiling...')
		self._compile()
		if verbose:
			self._summary()
	
	def _compile(self):
		self.classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy')

	def _summary(self):
		self.classifier.summary()
		logger.debug('Model built: {}'.format(pformat(self.classifier.get_config())))



# class MLP(ProsodicModel):

# 	def _prepare_params(self):
# 		self.prosodic_size = self.features.prosodic.nb_phones * self.features.prosodic.nb_features

# 	def build(self, filter_length=7, nb_hidden=100, mlp_activation='sigmoid', dropout_rate=0.5, verbose=True):
# 		if verbose:
# 			logger.info('Building...')

# 		inputs = []
# 		feats =	[]
# 		padding = filter_length // 2
# 		nb_filter = self.prosodic_size * filter_length
# 		cnn_activation = 'linear'
# 		cnn_init = 'one'

# 		sequence_pros = Input(name='input_pros', shape=(self.input_length, self.prosodic_size))
# 		inputs.append(sequence_pros)

# 		cnn1d_pad 		= ZeroPadding1D(padding)(sequence_pros)
# 		cnn1d 			= Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation=cnn_activation, init=cnn_init,
# 										subsample_length=1, border_mode='valid')(cnn1d_pad)

# 		hidden 			= TimeDistributed(Dense(output_dim=nb_hidden, activation=mlp_activation))(cnn1d)
# 		drop 			= Dropout(dropout_rate)(hidden)
# 		output 			= TimeDistributed(Dense(output_dim=self.nb_classes, activation='softmax'), name='output_source')(drop)
# 		self.classifier = Model(input=inputs, output=output)

# 		if verbose:
# 			logger.info('Compiling...')
# 		self._compile()
# 		if verbose:
# 			self._summary()
	
# 	def _compile(self):
# 		self.classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', sample_weight_mode='temporal')

# 	def _summary(self):
# 		self.classifier.summary()
# 		logger.debug('Model built: {}'.format(pformat(self.classifier.get_config())))