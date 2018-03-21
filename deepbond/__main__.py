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
	parser.add_argument('-r', '--split-ratio', default=0.6, type=float, help='ratio [0,1] to split the dataset into train/test (important only if k=1)')

	parser.add_argument('-t', '--train-strategy', default='bucket', type=str, help='strategy for training: bucket/one_per_batch/padding')
	parser.add_argument('-e', '--epochs', default=20, type=int, help='number of epochs to train the model')
	parser.add_argument('-b', '--batch-size', default=32, type=int, help='size of data batches to use for training, evaluating and predicting: 0 for buckets; 1 for one_per_batch; 2+ for padding')
	parser.add_argument('-k', '--kfold', default=5, type=int, help='number of folds to train the model')
	parser.add_argument('-v', '--val-split', default=0.0, type=float, help='ratio [0,1] to split the train dataset into train/validation (if 0 then alpha will be calculated using training data)')

	parser.add_argument('-d', '--dataset', type=str, help='one of: constituicao/constituicao_mini/pucrs_usp/pucrs_constituicao/controle/ccl/da')
	parser.add_argument('--task', type=str, default='ss', help='one of: ss/dd_fillers/dd_editdisfs/ssdd')
	parser.add_argument('--extra-data', action='store_true', help='add extra dataset as extension for training')

	parser.add_argument('--pos-type', type=str, default='nlpnet', help='pos tagger used for create POS features: nlpnet or probabilistc')
	parser.add_argument('--pos-file', type=str, default='data/resource/pos-pt/', help='dir or file for pos tagger pickled resources')
	parser.add_argument('--without-pos', action='store_true', help='do not use POS features')

	parser.add_argument('--emb-type', type=str, default='fonseca', help='method used for generate embeddings: complete list on embeddings.py')
	parser.add_argument('--emb-file', type=str, default='data/embeddings/fonseca/', help='dir or file for embeddings pickled resources')
	parser.add_argument('--without-emb', action='store_true', help='do not use POS embeddings')

	parser.add_argument('--use-handcrafted', action='store_true', help='use handcrafted features')

	parser.add_argument('--prosodic-type', type=str, default='principal', help='method used for select phones of a word')
	parser.add_argument('--prosodic-classify', action='store_true', help='classify prosodic info according to consonants')

	parser.add_argument('--models', nargs='+', type=str, default='rcnn rcnn')

	parser.add_argument('--save-predictions', type=str, default=None, help='save predictions in a file')

	parser.add_argument('--model-dir', type=str, default=model_dir, metavar='DIR', help='change directory where to save/load data, model, log, etc.')
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
	logger.debug('Analyzing errors for test data: {}'.format(options.dataset))
	gold_dir = get_path_from_dataset(options.dataset)['text_dir']
	pred_dir = 'data/saves/'+options.save_predictions
	if options.task == 'ss':
		ea = ErrorAnalysisSS(gold_dir=gold_dir, pred_dir=pred_dir)
		ea.most_frequent(k=10)
		ea.ngram_importance(n=[1, 2, 3], k=10)
		ea.average_sentence_length()
	
	elif options.task == 'dd_fillers':
		ea = ErrorAnalysisFillers(gold_dir=gold_dir, pred_dir=pred_dir)
		ea.most_frequent(k=10)
		ea.ngram_importance(n=[1, 2, 3], k=10)

	elif options.task == 'dd_editdisfs' or options.task == 'dd_editdisfs_binary':
		ea = ErrorAnalysisEditDisfs(gold_dir=gold_dir, pred_dir=pred_dir)
		ea.most_frequent(k=10)
		ea.ngram_importance(n=[1, 2, 3], k=10)


def run(options):
	'''Run deepbond :-)'''
	logger.debug('Running with options:\n{}'.format(options))

	# prepare train vocabulary
	logger.info('Loading vocabulary...')
	vocabulary = options.model_dir+'vocabulary.txt' if options.load else None

	# load data
	dsm 	 = load_dataset(options.dataset, extra=options.extra_data, vocabulary=vocabulary, task=options.task)
	features = load_features(options.pos_type, options.pos_file, options.emb_type, options.emb_file, 
							options.prosodic_type, options.prosodic_classify, 
							not options.without_pos, not options.without_emb, options.use_handcrafted)
	strategy = load_strategy(options.train_strategy, options.window_size, dsm.max_sentence_size)
	lexical_model, lexical_params, prosodic_model, prosodic_params = load_models(options.models[0], options.models[1], features, dsm.vocabulary, dsm.nb_classes, strategy)
	
	# train
	tm = TrainManager(lexical_model, lexical_params, prosodic_model, prosodic_params, strategy, options.batch_size, options.task)
	
	# load trained model
	if options.load:
		load(features, strategy, tm, options.model_dir)

	# embeddings info
	features.embeddings_statistics(dsm.get_texts())

	# k > 1: run k-fold cv
	# k == 1: run a single fold
	# k == 0: use loaded data to predict test data
	if options.kfold > 1:
		cv = CrossValidation(dsm, tm, options)
		cv.run(verbose=True)
	else:
		if options.kfold == 1:
			logger.info('Train/Test/Val summary: ')
			ds_train, ds_test = dsm.split(ratio=options.split_ratio, shuffle=False)
			ds_train.shuffle()
			ds_train.info()
			ds_test.info()
			ds_val = None
			if options.val_split > 0:
				ds_val = ds_train.truncate(options.val_split)
				ds_val.info()
			tm.train(ds_train, ds_test, ds_val, nb_epoch=options.epochs, verbose=True)
			if options.save_predictions:
				predictions = tm.get_test_predictions()
		elif options.kfold == 0:
			logger.info('Train/Test summary: ')
			ds_train, ds_test = dsm.split(ratio=0, shuffle=False)
			ds_train.info()
			ds_test.info()
			predictions = tm.evaluate(ds_test, verbose=True, vary_p=True)
		if options.save_predictions:
			fname = '%s-%s-%s-%s-%s-fold_0' % (*options.models, options.dataset, options.pos_type, options.emb_type)
			dname = options.save_predictions
			save_to_file(ds_test.word_texts, ds_test.shuffle_indexes, predictions, fname=fname, dname=dname, task=options.task)

	# report
	if options.save_predictions:
		analyze(options)

	# save model
	if options.save:
		save(features, dsm, strategy, tm, options.model_dir)	


def cli():
	'''Add some useful functionality here or import from a submodule'''

	# load the argument options
	options = load_options()

	# configure root logger to print to STDERR
	logger = logging.getLogger(__name__)
	root_logger = configure_stream(level='DEBUG')
	log_formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s]  %(message)s')
	file_handler = logging.FileHandler('{}.log'.format('data/log/'+options.id))
	file_handler.setFormatter(log_formatter)
	root_logger.addHandler(file_handler)

	# use GPU?
	# if options.gpu:
	# 	import theano.sandbox.cuda
	# 	theano.sandbox.cuda.use('gpu')

	if not os.path.exists(options.model_dir):
		os.makedirs(options.model_dir)

	run(options)



def detect(texts=[], audios=[], options={}, reset_vocabulary=False):

	if not os.path.exists(options['model_dir']):
		os.makedirs(options['model_dir'])

	print('Running with options:')
	print(options)

	print('Setting vocabulary...')
	vocabulary = os.path.join(options['model_dir'], 'vocabulary.txt')

	print('Building dataset from texts and audios...')
	dsm = build_dataset_from_data(texts, audios, vocabulary=vocabulary, task=options['task'], reset_vocabulary=reset_vocabulary)
	
	print('Setting features...')
	features = load_features(options['pos_type'], options['pos_file'], options['emb_type'], options['emb_file'], 
							options['prosodic_type'], options['prosodic_classify'], 
							not options['without_pos'], not options['without_emb'], options['use_handcrafted'])
	
	print('Setting model strategy...')
	strategy = load_strategy(options['train_strategy'], options['window_size'], dsm.max_sentence_size)
	
	print('Setting prediction model...')
	lexical_model, lexical_params, prosodic_model, prosodic_params = load_models(options['models'][0], options['models'][1], features, dsm.vocabulary, dsm.nb_classes, strategy)
	tm = TrainManager(lexical_model, lexical_params, prosodic_model, prosodic_params, strategy, options['batch_size'], options['task'])
	
	print('Loading model...')
	load(features, strategy, tm, options['model_dir'], verbose=False)

	print('Predicting...')
	_, ds_pred = dsm.split(ratio=0, shuffle=False)
	predictions = tm.evaluate(ds_pred, verbose=False, vary_p=False)
	
	print('Getting predictions...')
	pred_texts = get_new_texts(ds_pred.word_texts, ds_pred.shuffle_indexes, predictions, task=options['task'])

	return convert_predictions_to_tuples(pred_texts)


def set_options_manual(options):
	options_ss = dict(zip(options.keys(), options.values()))
	options_fillers = dict(zip(options.keys(), options.values()))
	options_editdisfs = dict(zip(options.keys(), options.values()))

	options_ss['task'] = 'ss'
	options_ss['id'] = 'SS_TEXT_CINDERELA'
	options_ss['model_dir'] = os.path.join('data/models/', options_ss['id'])+'/'

	options_fillers['task'] = 'dd_fillers'
	options_fillers['id'] = 'FILLERS_TEXT_CINDERELA'
	options_fillers['model_dir'] = os.path.join('data/models/', options_fillers['id'])+'/'

	options_editdisfs['task'] = 'dd_editdisfs_binary'
	options_editdisfs['id'] = 'EDITDISFS_TEXT_CINDERELA'
	options_editdisfs['model_dir'] = os.path.join('data/models/', options_editdisfs['id'])+'/'

	return options_ss, options_fillers, options_editdisfs


def remove_fillers_listbased(preds, list_fname):
	def load_list(fname):
		l = []
		with open(fname, 'r', encoding='utf8') as f:
			for line in f:
				l.append(line.strip())		
		return l
	filler_list = load_list(list_fname)
	new_preds = []
	for pred in preds:
		inner_preds = []
		for word, label in pred:
			if word not in filler_list:
				inner_preds.append((word, label))
		new_preds.append(inner_preds)
	return new_preds


def merge_ss_and_fillers(pred_ss, pred_fillers):
	new_preds = []
	for i in range(len(pred_ss)):
		inner_preds = []
		for t_ss, t_f in zip(pred_ss[i], pred_fillers[i]):
			
			if t_ss[0] != t_f[0]:
				print(t_ss[0], t_f[0])
				print(pred_ss[i]) 
				print(pred_fillers[i])
				assert(t_ss[0] == t_f[0])

			word = t_ss[0]
			label_ss = t_ss[1]
			label_f = t_f[1]
			if label_ss == '' and label_f == '':
				inner_preds.append((word, ''))
			elif label_ss != '' and label_f == '':
				inner_preds.append((word, label_ss))
			elif label_ss == '' and label_f != '':
				continue
			elif label_ss != '' and label_f != '':
				if inner_preds[-1][1] == '':
					inner_preds[-1] = (inner_preds[-1][0], label_ss)
		new_preds.append(inner_preds)
	return new_preds


def remove_disfs(pred_disfs):
	new_preds = []
	for preds in pred_disfs:
		inner_preds = []
		for word, label in preds:
			if label == '' or label == '.':
				inner_preds.append((word, label))
		new_preds.append(inner_preds)
	return new_preds


def pipeline(texts, audios, options):
	options_ss, options_fillers, options_editdisfs = set_options_manual(options)

	# sentence segmentation
	pred_ss = detect(texts, audios, options_ss)

	# filler detection (md + é)
	pred_fillers = detect(texts, [], options_fillers, reset_vocabulary=True)
	
	# merge sentence boundaries and fillers predictions
	new_preds = merge_ss_and_fillers(pred_ss, pred_fillers)

	# detect filled pauses using a list of selected words
	new_preds = remove_fillers_listbased(new_preds, 'data/lists/pp.txt')
	
	# convert predictions to texts
	new_texts = [' '.join(list(map(lambda x:x[0]+' '+x[1], text))) for text in new_preds]
	new_texts = [re.sub(r'\ +', ' ', text).strip() for text in new_texts]

	# detect edit disfluences
	pred_editdisfs = detect(new_texts, [], options_editdisfs, reset_vocabulary=True)

	# remove edit disfluences
	pred_editdisfs = remove_disfs(pred_editdisfs)

	# convert predictions to texts
	final_text = [' '.join(list(map(lambda x:x[0]+x[1], text))) for text in pred_editdisfs]

	return final_text


if __name__ == '__main__':
	# cli()

	texts = ['ela morava com a madrasta as irmã né e ela era diferenciada das três era maltratada ela tinha que fazer limpeza na casa toda no castelo alias e as irmãs não faziam nada',
			'aqui é uma menininha simples uhn humilde eu creio creio que era humilde tava com os pais vivia com os pais depois é ela tinha é esse cavalo esse animalzinho de estimação depois ela foi morar no palácio',
			'era uma vez uma uma menina uma garota né que vivia numa castelo com o pai e ela gostava muito de animais e ela estava ahn fazendo um passeio a cavalo e ela morava num castelo']
	audios = []
	# audios = ['data/prosodic/CCL-A/2.csv', 'data/prosodic/CCL-A/3.csv', 'data/prosodic/Controle/23.csv']

	options = vars(load_options())
	preds = pipeline(texts, audios, options)
