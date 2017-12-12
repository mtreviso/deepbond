import numpy as np

import logging
from pprint import pformat

from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import *

from deepbond.models.lexical import LexicalModel

logger = logging.getLogger(__name__)


class MLP(LexicalModel):

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

	def build(self, nb_hidden=100, mlp_activation='sigmoid', dropout_rate=0.5, verbose=True):
		if verbose:
			logger.info('Building...')

		inputs = []
		feats =	[]

		if self.use_embeddings:
			sequence = Input(name='input_emb', shape=(self.input_length,), dtype='int32')
			embedded = Embedding(self.emb_vocab_size, self.emb_size, input_length=self.input_length, weights=[self.emb_weights])(sequence)
			inputs.append(sequence)
			feats.append(embedded)
			merge_features = embedded

		if self.use_pos:
			sequence_pos = Input(name='input_pos', shape=(self.input_length,), dtype='int32')
			embedded_pos = Embedding(self.POS_vocab_size, self.POS_emb_size, input_length=self.input_length, init='glorot_normal')(sequence_pos)
			inputs.append(sequence_pos)
			feats.append(embedded_pos)
			merge_features = embedded_pos

		if self.use_handcrafted:
			sequence_hc = Input(name='input_hc', shape=(self.input_length, self.hc_size))
			sequence_hc_x = TimeDistributed(Dense(nb_hidden // 4, activation='relu'))(sequence_hc)
			inputs.append(sequence_hc)
			feats.append(sequence_hc_x)
			merge_features = sequence_hc_x
		
		if sum([self.use_embeddings, self.use_pos, self.use_handcrafted]) > 1:
			merge_features 	= merge(feats, mode='concat', concat_axis=-1)

		flattened 		= Flatten()(merge_features)
		hidden 			= Dense(output_dim=nb_hidden, activation='relu')(flattened)
		drop 			= Dropout(dropout_rate)(hidden)
		output 			= Dense(output_dim=self.nb_classes, activation='softmax', name='output_source')(drop)
		self.classifier = Model(input=inputs, output=output)

		if verbose:
			logger.info('Compiling...')
		self._compile()
		if verbose:
			self._summary()
	
	def _compile(self):
		# optimizer = RMSprop(lr=1e-4)
		# self.classifier.compile(optimizer=optimizer, loss='categorical_crossentropy')
		self.classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy')

	def _summary(self):
		self.classifier.summary()
		logger.debug('Model built: {}'.format(pformat(self.classifier.get_config())))

