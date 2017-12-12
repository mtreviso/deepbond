import numpy as np
import logging

from keras import backend as K
from keras.models import Model
from keras.layers import *

from deepbond.statistics import Statistics
from deepbond.models import WindowEnsembleMLP

logger = logging.getLogger(__name__)


class WindowEnsembleStateful(WindowEnsembleMLP):
	'''
	important: batch_size must be equal to 1
	'''

	def build(self):
		logger.info('Building...')

		# text model
		sequence_text = Input(name='input_source', shape=(self.input_length,), dtype='int32', batch_shape=(self.batch_size, self.input_length))
		embedded_text = Embedding(self.vocabulary_size, self.embeddings_size, input_length=self.input_length, weights=[self.embeddings_weights])(sequence_text)
		sequence_pos = Input(name='input_pos', shape=(self.input_length,), dtype='int32', batch_shape=(self.batch_size, self.input_length))
		embedded_pos = Embedding(self.POS_vocab_size, self.POS_embeddings_size, input_length=self.input_length, init='glorot_normal')(sequence_pos)
		merge_embedded = merge([embedded_text, embedded_pos], mode='concat', concat_axis=-1)
		text_f = GRU(self.nb_hidden, return_sequences=True, stateful=True, activation='sigmoid')(merge_embedded)
		text_b = GRU(self.nb_hidden, return_sequences=True, stateful=True, go_backwards=True, activation='sigmoid')(merge_embedded)
		text_sum = merge([text_f, text_b], mode='sum', concat_axis=-1)
		text_drop = Dropout(self.dropout_rate)(text_sum)
		text_flattened = Flatten()(text_drop)
		text_output = Dense(self.nb_classes, activation='softmax')(text_flattened)
		self.text_model = Model(input=[sequence_text, sequence_pos], output=text_output)

		# audio model
		self.prosodic_size = self.features.prosodic_nb_phones * self.features.prosodic_nb_features
		sequence_audio = Input(name='input_audio', shape=(self.input_length, self.prosodic_size), batch_shape=(self.batch_size, self.input_length, self.prosodic_size))
		audio_f = GRU(self.nb_hidden, return_sequences=True, stateful=True, activation='sigmoid')(sequence_audio)
		audio_b = GRU(self.nb_hidden, return_sequences=True, stateful=True, go_backwards=True, activation='sigmoid')(sequence_audio)
		audio_sum = merge([audio_f, audio_b], mode='sum', concat_axis=-1)
		audio_drop = Dropout(self.dropout_rate)(audio_sum)
		audio_flattened = Flatten()(audio_drop)
		audio_output = Dense(self.nb_classes, activation='softmax')(audio_flattened)
		self.audio_model = Model(input=[sequence_audio], output=audio_output)

		self._compile()
