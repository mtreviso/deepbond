import numpy as np

import logging
from pprint import pformat

from keras.models import Model
from keras.layers import *

from deepbond.models.prosodic import ProsodicModel

logger = logging.getLogger(__name__)


class RecNN(ProsodicModel):

	def _prepare_params(self):
		self.prosodic_size = self.features.prosodic.nb_phones * self.features.prosodic.nb_features

	def build(self, nb_hidden=100, rnn='LSTM', rnn_activation='sigmoid', dropout_rate=0.5, verbose=True):
		if verbose:
			logger.info('Building...')

		RNN = LSTM if rnn == 'LSTM' else GRU
		inputs = []
		feats =	[]

		sequence_pros = Input(name='input_pros', shape=(self.input_length, self.prosodic_size))
		inputs.append(sequence_pros)

		forward_rnn 	= RNN(nb_hidden, return_sequences=True, activation=rnn_activation)(sequence_pros)
		backward_rnn 	= RNN(nb_hidden, return_sequences=True, go_backwards=True, activation=rnn_activation)(sequence_pros)
		merge_rnn 		= merge([forward_rnn, backward_rnn], mode='sum', concat_axis=-1)
		drop 			= Dropout(dropout_rate)(merge_rnn)
		output 			= TimeDistributed(Dense(output_dim=self.nb_classes, activation='softmax'), name='output_source')(drop)
		self.classifier = Model(input=inputs, output=output)

		if verbose:
			logger.info('Compiling...')
		self._compile()
		if verbose:
			self._summary()
	
	def _compile(self):
		self.classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', sample_weight_mode='temporal')

	def _summary(self):
		self.classifier.summary()
		logger.debug('Model built: {}'.format(pformat(self.classifier.get_config())))