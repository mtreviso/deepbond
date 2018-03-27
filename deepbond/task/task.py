import os, re
import json
from deepbond.dataset import DataSetManager
from deepbond.features import Features
import deepbond.models.strategies as Strategies
from deepbond.loader import build_dataset_from_data, load_models
from deepbond.models.manager import TrainManager
from deepbond.models.utils import get_new_texts, convert_predictions_to_tuples
from deepbond.train import load, save


class Task:

	def __init__(self, l_model='rcnn', p_model='rcnn', verbose=True):
		self.l_model = l_model
		self.p_model = p_model
		self.verbose = verbose
		self.options = self._get_default_options()
		self.set_model_dir(self.options['model_dir'])
		self.features = None
		self.strategy = None
		self.tm = None
		self.vocabulary = None
		self.nb_classes = None

	def _get_default_options(self):
		return {
			'id': 'default_example_ss_controle',	# all models will be stored in data/models/:id:
			'task': 'ss', 							# one of ss/dd_fillers/dd_editdisfs/ssdd
			'dataset': 'controle', 					# see loader.py (used only for training and error analysis)

			'load': True, 			# load the trained model for :id:
			'save': False, 			# save the trained model for :id:

			'window_size': 7, 		# size of the sliding window
			'split_ratio': 0.6,		# ratio [0,1] to split the dataset into train/test

			'train_strategy': 'bucket', # strategy for training: bucket/window/dicted/padding
			'epochs': 15,				# number of epochs for neural nets
			'batch_size': 32,			# size of data batches to use for training and predicting
			'kfold': 5,					# number of folds to evaluate the model
			'val_split': 0.0,			# ratio [0,1] to split the train dataset into train/validation

			'extra_data': False,		# see loader.py

			'pos_type': 'nlpnet',					# pos tagger: nlpnet or pickled pos tagger
			'pos_file': 'data/resource/pos-pt/', 	# dir or file for embeddings pickled resources
			'without_pos': True,					# do not use pos

			'emb_type': 'word2vec',		# method used for generate embeddings: see features/embeddings
			'emb_file': '',	# e.g. Embeddings-Deepbond/ptbr/word2vec/pt_word2vec_sg_600.emb
			'without_emb': False,		# do not use embeddings

			'use_handcrafted': False,	# do not use handcrafter features (useful only for edit disfluences)

			'prosodic_type': 'principal', 	# method used for select phones of a word
			'prosodic_classify': False,		# classify prosodic info according to consonants

			'models': 'rcnn none',		# lexical and prosodic (rcnn for lexical and none for prosodic)

			'save_predictions': None, 	# dirname to save cv predictions in data/saves/ (None means do not save)

			'model_dir': 'data/models/', # directory where data, model, log, etc. will be saved (data/models/:id:/)
			'gpu': True 				 # run on GPU instead of on CPU
		}

	def set_lexical_model(self, l_model):
		self.l_model = l_model

	def set_prosodic_model(self, p_model):
		self.p_model = p_model

	def set_model_dir(self, model_dir):
		self.options['model_dir'] = model_dir
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)

	def set_model_id(self, model_id):
		self.options['id'] = model_id
		self.set_model_dir(os.path.join('data/models/', model_id)+'/')

	def _print(self, txt):
		if self.verbose:
			print(txt)

	def load_features(self, filename=''):
		if filename == '':
			filename = os.path.join(self.options['model_dir'], 'features.json')
		with open(filename, 'r') as f:
			data = json.load(f)
		self.features = Features(**data)

	def load_strategy(self, filename=''):
		if filename == '':
			filename = os.path.join(self.options['model_dir'], 'strategy.json') 
		with open(filename, 'r') as f:
			data = json.load(f)
		if self.options['train_strategy'] == 'bucket':
			self.strategy = Strategies.BucketStrategy(**data)
		elif self.options['train_strategy'] == 'padding':
			self.strategy = Strategies.PaddingStrategy(**data)
		elif self.options['train_strategy'] == 'window':
			self.strategy = Strategies.WindowStrategy(**data)
		elif self.options['train_strategy'] == 'dicted':
			self.strategy = Strategies.DictedStrategy(**data)

	def load_vocabulary(self, filename=''):
		if filename == '':
			filename = os.path.join(self.options['model_dir'], 'vocabulary.txt') 
		DataSetManager.reset_vocabulary()
		self.vocabulary = DataSetManager.load_and_get_vocabulary(filename)
		self.nb_classes = DataSetManager.get_nb_classes(self.options['task'])

	def load_trained_model(self, dirname=''):
		if dirname == '':
			dirname = self.options['model_dir'] 

		models_and_params = load_models(self.l_model, self.p_model, 
										self.features, self.vocabulary, 
										self.nb_classes, self.strategy)
		
		self.tm = TrainManager(*models_and_params, self.strategy, 
								self.options['batch_size'], 
								self.options['task'])
		self.tm.load(dirname, verbose=False)

	def load(self):
		if self.features is None:
			self.load_features()
		if self.strategy is None:
			self.load_strategy()
		if self.vocabulary is None:
			self.load_vocabulary()
		if self.tm is None:
			self.load_trained_model()

	def detect(self, texts=[], audios=[]):
		self._print('Running with options: {}'.format(self.options))

		self._print('Loading model...')
		self.load()

		self._print('Building dataset from texts and audios...')
		dsm = build_dataset_from_data(texts, audios, task=self.options['task'])

		self._print('Predicting...')
		_, ds_pred = dsm.split(ratio=0, shuffle=False)
		predictions = self.tm.evaluate(ds_pred, verbose=False, vary_p=False)
		
		self._print('Getting predictions...')
		pred_texts = get_new_texts(ds_pred.word_texts, ds_pred.shuffle_indexes, predictions, task=self.options['task'])

		return convert_predictions_to_tuples(pred_texts)
