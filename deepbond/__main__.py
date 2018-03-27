#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
deepbond.__main__
~~~~~~~~~~~~~~~~~~~~~~~

The main entry point for the command line interface.

Invoke as ``deepbond`` (if installed)
or ``python -m deepbond --id YourModelName`` (no install required).
"""
from __future__ import absolute_import, unicode_literals
import argparse
import logging
import os, re

from deepbond import __prog__, __title__, __summary__, __uri__, __version__
from deepbond.log import configure_stream
from deepbond.loader import get_path_from_dataset, build_dataset_from_data, load_dataset, load_features, load_strategy, load_models
from deepbond.models.manager import TrainManager
from deepbond.models.cv import CrossValidation
from deepbond.models.utils import save_to_file, get_new_texts, convert_predictions_to_tuples, convert_tuples_to_texts

logger = logging.getLogger(__name__)


def load_options():
	'''
	Load the options from arguments 
	:return: argument_config
	'''

	config_arg_parser = argparse.ArgumentParser(add_help=False, prog=__prog__)
	config_arg_parser.add_argument('--version', action='version', version='%(prog)s {}'.format(__version__))
	config_arg_parser.add_argument('--id', required=True, default='DeepBondModel', help='name of configuration file in configs/folder')
	args, remaining_argv = config_arg_parser.parse_known_args()
	model_dir = os.path.join('data/models/', args.id)+'/'

	parser = argparse.ArgumentParser(description='{}: {}'.format(__title__, __summary__),
									 epilog='Please visit {} for additional help.'.format(__uri__),
									 parents=[config_arg_parser], add_help=True)

	parser.add_argument('-l', '--load', action='store_true', help='load the trained model for :id:')
	parser.add_argument('-s', '--save', action='store_true', help='save the trained model for :id:')

	parser.add_argument('-w', '--window-size', default=7, type=int, help='size of the sliding window')
	parser.add_argument('-r', '--split-ratio', default=0.6, type=float, help='ratio [0,1] to split the dataset into train/test')

	parser.add_argument('-t', '--train-strategy', default='bucket', type=str, help='strategy for training: bucket/one_per_batch/padding')
	parser.add_argument('-e', '--epochs', default=20, type=int, help='number of epochs to train the model')
	parser.add_argument('-b', '--batch-size', default=32, type=int, help='size of data batches to use for training and predicting')
	parser.add_argument('-k', '--kfold', default=5, type=int, help='number of folds to evaluate the model')
	parser.add_argument('-v', '--val-split', default=0.0, type=float, help='ratio [0,1] to split the train dataset into train/validation (if 0 then alpha will be calculated using training data)')

	parser.add_argument('-d', '--dataset', type=str, help='one of: constituicao/constituicao_mini/pucrs_usp/pucrs_constituicao/controle/ccl/da')
	parser.add_argument('--task', type=str, default='ss', help='one of: ss/dd_fillers/dd_editdisfs/ssdd')
	parser.add_argument('--extra-data', action='store_true', help='add extra dataset as extension for training')

	parser.add_argument('--pos-type', type=str, default='nlpnet', help='pos tagger used POS features: nlpnet or pickled pos tagger')
	parser.add_argument('--pos-file', type=str, default='data/resource/pos-pt/', help='dir or file for pos tagger pickled resources')
	parser.add_argument('--without-pos', action='store_true', help='do not use POS features')

	parser.add_argument('--emb-type', type=str, default='fonseca', help='method used for generate embeddings: complete list on embeddings.py')
	parser.add_argument('--emb-file', type=str, default='data/embeddings/fonseca/', help='dir or file for embeddings ,pdeçs')
	parser.add_argument('--without-emb', action='store_true', help='do not use embeddings')

	parser.add_argument('--use-handcrafted', action='store_true', help='use handcrafted features')

	parser.add_argument('--prosodic-type', type=str, default='principal', help='method used for select phones of a word')
	parser.add_argument('--prosodic-classify', action='store_true', help='classify prosodic info according to consonants')

	parser.add_argument('--models', nargs='+', type=str, default='rcnn rcnn')

	parser.add_argument('--save-predictions', type=str, default=None, help='dirname to save predictions in data/saves/')

	parser.add_argument('--model-dir', type=str, default=model_dir, metavar='DIR', help='directory where to save/load data, model, log, etc.')
	parser.add_argument('--gpu', action='store_true', help='run on GPU instead of on CPU')

	argument_config = parser.parse_args()
	return argument_config


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


def run(options):
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
	# k == 1: run a single fold
	# k == 0: use loaded data to predict test data
	if options['kfold'] > 1:
		cv = CrossValidation(dsm, tm, options)
		cv.run(verbose=True)
	else:
		if options['kfold'] == 1:
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
			if options['save_predictions']:
				predictions = tm.get_test_predictions()
		elif options['kfold'] == 0:
			logger.info('Train/Test summary: ')
			ds_train, ds_test = dsm.split(ratio=0, shuffle=False)
			ds_train.info()
			ds_test.info()
			predictions = tm.evaluate(ds_test, verbose=True, vary_p=True)
		if options['save_predictions']:
			fname = '%s-%s-%s-%s-%s-fold_0' % (*options['models'], options['dataset'], options['pos_type'], options['emb_type'])
			dname = options['save_predictions']
			save_to_file(ds_test.word_texts, ds_test.shuffle_indexes, predictions, fname=fname, dname=dname, task=options['task'])

	# report
	if options['save_predictions']:
		analyze(options)

	# save model
	if options['save']:
		save(features, dsm, strategy, tm, options['model_dir'])	


def cli():
	'''Add some useful functionality here or import from a submodule'''

	# load the argument options
	options = vars(load_options())

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

	run(options)


if __name__ == '__main__':
	cli()

