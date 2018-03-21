import os, re
from deepbond.loader import build_dataset_from_data, load_features, load_strategy, load_models
from deepbond.models.manager import TrainManager
from deepbond.models.utils import get_new_texts, convert_predictions_to_tuples
from deepbond.train import load, save


def detect(texts=[], audios=[], options={}, reset_vocabulary=False, verbose=True):

	if not os.path.exists(options['model_dir']):
		os.makedirs(options['model_dir'])

	if verbose:
		print('Running with options:')
		print(options)

	if verbose:
		print('Setting vocabulary...')
	vocabulary = os.path.join(options['model_dir'], 'vocabulary.txt')

	if verbose:
		print('Building dataset from texts and audios...')
	dsm = build_dataset_from_data(texts, audios, vocabulary=vocabulary, task=options['task'], reset_vocabulary=reset_vocabulary)
	
	if verbose:
		print('Setting features...')
	features = load_features(options['pos_type'], options['pos_file'], options['emb_type'], options['emb_file'], 
							options['prosodic_type'], options['prosodic_classify'], 
							not options['without_pos'], not options['without_emb'], options['use_handcrafted'])
	
	if verbose:
		print('Setting model strategy...')
	strategy = load_strategy(options['train_strategy'], options['window_size'], dsm.max_sentence_size)
	
	if verbose:
		print('Setting prediction model...')
	lexical_model, lexical_params, prosodic_model, prosodic_params = load_models(options['models'][0], options['models'][1], features, dsm.vocabulary, dsm.nb_classes, strategy)
	tm = TrainManager(lexical_model, lexical_params, prosodic_model, prosodic_params, strategy, options['batch_size'], options['task'])
	
	if verbose:
		print('Loading model...')
	load(features, strategy, tm, options['model_dir'], verbose=False)

	if verbose:
		print('Predicting...')
	_, ds_pred = dsm.split(ratio=0, shuffle=False)
	predictions = tm.evaluate(ds_pred, verbose=False, vary_p=False)
	
	if verbose:
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


def pipeline(texts, audios, options_ss, options_fillers, options_editdisfs, verbose=True):

	# sentence segmentation
	pred_ss = detect(texts, audios, options_ss, verbose=verbose)

	# filler detection (md + é)
	pred_fillers = detect(texts, [], options_fillers, reset_vocabulary=True, verbose=verbose)
	
	# merge sentence boundaries and fillers predictions
	new_preds = merge_ss_and_fillers(pred_ss, pred_fillers)

	# detect filled pauses using a list of selected words
	new_preds = remove_fillers_listbased(new_preds, 'data/lists/pp.txt')
	
	# convert predictions to texts
	new_texts = [' '.join(list(map(lambda x:x[0]+' '+x[1], text))) for text in new_preds]
	new_texts = [re.sub(r'\ +', ' ', text).strip() for text in new_texts]

	# detect edit disfluences
	pred_editdisfs = detect(new_texts, [], options_editdisfs, reset_vocabulary=True, verbose=verbose)

	# remove edit disfluences
	pred_editdisfs = remove_disfs(pred_editdisfs)

	# convert predictions to texts
	final_text = [' '.join(list(map(lambda x:x[0]+x[1], text))) for text in pred_editdisfs]

	return final_text


def get_default_options():
	return {
		# required!
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