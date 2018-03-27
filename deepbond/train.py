from __future__ import absolute_import, unicode_literals
import argparse
import logging
import os, re

from deepbond import __prog__, __title__, __summary__, __uri__, __version__
from deepbond.log import configure_stream
from deepbond.loader import get_path_from_dataset, load_dataset, load_features, load_strategy, load_models
from deepbond.models.manager import TrainManager
from deepbond.models.cv import CrossValidation
from deepbond.models.utils import save_to_file

logger = logging.getLogger(__name__)


def load(features, strategy, tm, model_dir, verbose=True):
	logger.info('Loading features info...')
	features.load(model_dir+'features.json')

	logger.info('Loading strategy info...')
	strategy.load(model_dir+'strategy.json')

	logger.info('Loading models weights...')
	tm.load(model_dir, verbose=verbose)


def save(features, dsm, strategy, tm, model_dir):
	logger.info('Saving features info...')
	features.save(model_dir+'features.json')

	logger.info('Saving vocabulary...')
	dsm.save_vocabulary(model_dir+'vocabulary.txt')

	logger.info('Saving strategy info...')
	strategy.save(model_dir+'strategy.json')

	logger.info('Saving models weights...')
	tm.save(model_dir)


def analyze(options):
	from deepbond.error_analysis import ErrorAnalysisSS, ErrorAnalysisFillers, ErrorAnalysisEditDisfs
	logger.debug('Analyzing errors for test data: {}'.format(options['dataset']))
	gold_dir = get_path_from_dataset(options['dataset'])['text_dir']
	pred_dir = 'data/saves/'+options['save_predictions']
	if options['task'] == 'ss':
		ea = ErrorAnalysisSS(gold_dir=gold_dir, pred_dir=pred_dir)
		ea.most_frequent(k=10)
		ea.ngram_importance(n=[1, 2, 3], k=10)
		ea.average_sentence_length()
	
	elif options['task'] == 'dd_fillers':
		ea = ErrorAnalysisFillers(gold_dir=gold_dir, pred_dir=pred_dir)
		ea.most_frequent(k=10)
		ea.ngram_importance(n=[1, 2, 3], k=10)

	elif options['task'] == 'dd_editdisfs' or options['task'] == 'dd_editdisfs_binary':
		ea = ErrorAnalysisEditDisfs(gold_dir=gold_dir, pred_dir=pred_dir)
		ea.most_frequent(k=10)
		ea.ngram_importance(n=[1, 2, 3], k=10)


def train(options):
	'''Run deepbond :-)'''
	logger.debug('Running with options:\n{}'.format(options))

	# prepare train vocabulary
	logger.info('Loading vocabulary...')
	vocabulary = options['model_dir']+'vocabulary.txt' if options['load'] else None

	# load data
	dsm 	 = load_dataset(options['dataset'], extra=options['extra_data'], vocabulary=vocabulary, task=options['task'])
	features = load_features(options['pos_type'], options['pos_file'], options['emb_type'], options['emb_file'], 
							options['prosodic_type'], options['prosodic_classify'], 
							not options['without_pos'], not options['without_emb'], options['use_handcrafted'])
	strategy = load_strategy(options['train_strategy'], options['window_size'], dsm.max_sentence_size)
	lexical_model, lexical_params, prosodic_model, prosodic_params = load_models(options['models'][0], options['models'][1], features, dsm.vocabulary, dsm.nb_classes, strategy)
	
	# train
	tm = TrainManager(lexical_model, lexical_params, prosodic_model, prosodic_params, strategy, options['batch_size'], options['task'])
	
	# load trained model
	if options['load']:
		load(features, strategy, tm, options['model_dir'])

	# embeddings info
	features.embeddings_statistics(dsm.get_texts())

	# k > 1: run k-fold cv
	# k == 1: run a single fold with val split
	if options['kfold'] > 1:
		cv = CrossValidation(dsm, tm, options)
		cv.run(verbose=True)
	elif options['kfold'] == 1:
		logger.info('Train/Test/Val summary: ')
		ds_train, ds_test = dsm.split(ratio=options['split_ratio'], shuffle=False)
		ds_train.shuffle()
		ds_train.info()
		ds_test.info()
		ds_val = None
		if options['val_split'] > 0:
			ds_val = ds_train.truncate(options['val_split'])
			ds_val.info()
		tm.train(ds_train, ds_test, ds_val, nb_epoch=options['epochs'], verbose=True)
	
	# save predictions in txt 
	if options['save_predictions']:
		predictions = tm.get_test_predictions()
		fname = '%s-%s-%s-%s-%s-fold_0' % (*options['models'], options['dataset'], options['pos_type'], options['emb_type'])
		dname = options['save_predictions']
		save_to_file(ds_test.word_texts, ds_test.shuffle_indexes, predictions, fname=fname, dname=dname, task=options['task'])

	# report error analysis
	if options['save_predictions']:
		analyze(options)

	# save model
	if options['save']:
		save(features, dsm, strategy, tm, options['model_dir'])	


def get_default_options():
	return {
		# required!
		'id': 'default_example_ss_controle',	# all models will be stored in data/models/:id:
		'task': 'ss', 							# one of ss/dd_fillers/dd_editdisfs/ssdd
		'dataset': 'controle', 					# see loader.py (used only for training and error analysis)

		'load': False, 			# load the trained model for :id:
		'save': True, 			# save the trained model for :id:

		'window_size': 7, 		# size of the sliding window
		'split_ratio': 1.0,		# ratio [0,1] to split the dataset into train/test

		'train_strategy': 'bucket', # strategy for training: bucket/window/dicted/padding
		'epochs': 15,				# number of epochs for neural nets
		'batch_size': 32,			# size of data batches to use for training and predicting
		'kfold': 1,					# number of folds to evaluate the model
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

		'models': ['rcnn', 'none'],		# lexical and prosodic (rcnn for lexical and none for prosodic)

		'save_predictions': None, 	# dirname to save cv predictions in data/saves/ (None means do not save)

		'model_dir': 'data/models/', # directory where data, model, log, etc. will be saved (data/models/:id:/)
		'gpu': True 				 # run on GPU instead of on CPU
	}


def configure(options={}):
	'''Add some useful functionality here or import from a submodule'''

	if options == {}:
		# load the argument options
		options = get_default_options()

	# configure root logger to print to STDERR
	logger = logging.getLogger(__name__)
	root_logger = configure_stream(level='DEBUG')
	log_formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s]  %(message)s')
	file_handler = logging.FileHandler('{}.log'.format('data/log/'+options['id']))
	file_handler.setFormatter(log_formatter)
	root_logger.addHandler(file_handler)

	# use GPU?
	# if options['gpu']:
	# 	import theano.sandbox.cuda
	# 	theano.sandbox.cuda.use('gpu')

	if not os.path.exists(options['model_dir']):
		os.makedirs(options['model_dir'])

