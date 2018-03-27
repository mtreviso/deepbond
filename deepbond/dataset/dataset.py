import logging
from abc import ABCMeta, abstractmethod
from os import listdir
from numpy.random import shuffle as shuffle_list
from deepbond.dataset import SentTokenizer

logger = logging.getLogger(__name__)
tokenizer = SentTokenizer() # vocabulary

class DataSet(metaclass=ABCMeta):

	def __init__(self, text_dir=None, prosodic_file=None, vocabulary=None):
		"""
		Class to deal with an annotated corpus
		:param text_dir: a path to text files (.txt)
		:param prosodic_file: a path to a prosodic file (.csv)
		"""
		if text_dir is None and prosodic_file is None:
			raise Exception('You should inform a text directory or a prosodic file!')

		if vocabulary is not None and not tokenizer.loaded:
			tokenizer.load_vocabulary(vocabulary)
		self.texts = []
		self.word_texts = []
		self.word_id_texts = []
		self.pros_texts = []
		if text_dir:
			self._read_text_dir(text_dir)
			self._fit_tokenizer()
			self._set_nb_attributes()
		if prosodic_file:
			self._read_pros_file(prosodic_file)

	def _read_text_dir(self, dname):
		slash = '' if dname[-1] == '/' else '/'
		for fname in sorted(listdir(dname)):
			try:
				self._read_text_file(dname + slash + fname)
				# self._save_cleaned_file(dname + slash + fname)
			except:
				pass
	
	def _read_text_file(self, filename):
		f = open(filename, 'r', encoding='utf8')
		txt = self._clean_text_file(f.read().strip())
		tks = txt.strip().split()
		min_sentence_size = 5
		if txt.strip() and len(tks) >= min_sentence_size and tks[0] not in '.!?:;':
			self.texts.append(txt)
		else:
			print(txt)
		f.close()

	def _save_cleaned_file(self, filename):
		if 'ABCD' in filename or 'PUCRS' in filename:
			f = open(filename, 'w', encoding='utf8')
			f.write(self.texts[-1])
			f.close()

	@abstractmethod
	def _clean_text_file(self, text):
		"""
		should return a text with only one line and . as punctation mark
		"""
		pass

	def _fit_tokenizer(self, build_vocab=True):
		if not self.texts:
			raise Exception('You should read the data before fit the tokenizer!')
		self.word_texts = tokenizer.fit_on_texts(self.texts, build_vocab=build_vocab)
		self.word_id_texts = tokenizer.word_texts_to_indexes(self.word_texts)
		self.pros_texts = [[] for _ in self.word_id_texts]
		self.vocabulary = tokenizer.word_index

	def _set_nb_attributes(self):
		self.nb_texts = len(self.word_texts)
		self.nb_sentences = sum(map(lambda x: x.count('.'), self.word_texts))
		self.nb_words = sum(map(len, self.word_texts))
		self.max_sentence_size = max(map(lambda x: len(x)-x.count('.'), self.word_texts))
		self.nb_classes = len(tokenizer.labels)
		self.nb_disfs = sum(map(lambda x: x.count('*')+x.count('+')+x.count('$'), self.word_texts))
		self.vocab_size = len(tokenizer.word_index)

	def indexes_to_words(self, indexes_texts):
		return tokenizer.index_texts_to_text(indexes_texts)

	def _read_pros_file(self, filename):
		import pandas as pd
		pros = pd.read_csv(filename, encoding='utf8')
		pros = pros.drop(['n interval', 'Start', 'Finish', 'fo1', 'fo3'], axis=1)
		pros = pros.replace('--undefined--', 0)
		pros = pros.where((pd.notnull(pros)), None) # transform NaN to None
		# normalize per speaker too
		scale_minmax = lambda x: (x - x.min()) / (x.max() - x.min())
		pros['Intensity'] = scale_minmax(pros['Intensity'].astype(float))
		pros['fo2'] = scale_minmax(pros['fo2'].astype(float))
		pros['Duration'] = pros['Duration'] / 1000
		self.pros_texts = list(pros.groupby('Filename'))

	def info(self):
		logger.info(type(self).__name__)
		logger.info('Nb texts: {}'.format(self.nb_texts))
		logger.info('Nb sentences: {}'.format(self.nb_sentences))
		logger.info('Nb words: {}'.format(self.nb_words))
		logger.info('Nb disfs: {}'.format(self.nb_disfs))
		logger.info('Nb classes: {}'.format(self.nb_words))

	
class DataSetManager:

	def __init__(self, originals=[], extensions=[], task='ss'):
		"""
		Class to deal with a collection of datasets
		:param originals: list of datasets
		:param extensions: list of datasets
		"""
		self.originals = originals
		self.extensions = extensions
		self.task = task
		self.word_texts = None
		self._set_nb_attributes()

	@property
	def vocabulary(self):
		return tokenizer.word_index

	@property
	def nb_classes(self):
		return DataSetManager.get_nb_classes()

	@staticmethod
	def get_nb_classes(task):
		if task == 'ss':
			return 2
		elif task == 'dd_fillers':
			return 3 
		elif task == 'dd_editdisfs':
			return 4 
		elif task == 'dd_editdisfs_binary':
			return 2 
		elif task == 'ssdd':
			return 2
		return 2

	@staticmethod 
	def load_and_get_vocabulary(filename):
		tokenizer.load_vocabulary(filename)
		return tokenizer.word_index

	@staticmethod 
	def reset_vocabulary():
		tokenizer.__init__()

	def get_texts(self):
		if self.word_texts is not None:
			return self.word_texts
		self.word_texts = []
		for ds in self.originals:
			self.word_texts.extend(ds.word_texts)
		for ds in self.extensions:
			self.word_texts.extend(ds.word_texts)
		return self.word_texts

	def save_vocabulary(self, filename):
		tokenizer.save_vocabulary(filename)

	def load_vocabulary(self, filename):
		tokenizer.load_vocabulary(filename)

	def _set_nb_attributes(self):
		self.nb_texts = 0
		self.nb_sentences = 0
		self.nb_words = 0
		self.max_sentence_size = 0
		self.nb_disfs = 0
		for ds in self.originals + self.extensions:
			self.nb_texts += ds.nb_texts
			self.nb_sentences += ds.nb_sentences
			self.nb_words += ds.nb_words
			self.nb_disfs += ds.nb_disfs
			self.max_sentence_size = max(ds.max_sentence_size, self.max_sentence_size)

	def info(self):
		logger.info(type(self).__name__)
		logger.info('Nb texts: {}'.format(self.nb_texts))
		logger.info('Nb sentences: {}'.format(self.nb_sentences))
		logger.info('Nb words: {}'.format(self.nb_words))
		logger.info('Nb disfs: {}'.format(self.nb_disfs))
		logger.info('Vocabulary size: {}'.format(len(self.vocabulary)))
		logger.info('Avg sentence size: {}'.format(self.nb_words / self.nb_sentences))
		logger.info('Avg sentence per file: {}'.format(self.nb_sentences / self.nb_texts))

	def split(self, ratio=0.8, shuffle=True, olds=[]):
		if self.task == 'ss':
			ds_train = DataSetSS()
			ds_test = DataSetSS()
		elif self.task == 'dd_fillers':
			ds_train = DataSetFillers()
			ds_test = DataSetFillers()
		elif self.task == 'dd_editdisfs':
			ds_train = DataSetEditDisfs()
			ds_test = DataSetEditDisfs()
		elif self.task == 'dd_editdisfs_binary':
			ds_train = DataSetEditDisfs(binary=True)
			ds_test = DataSetEditDisfs(binary=True)
		else:
			ds_train = DataSetSSandDD()
			ds_test = DataSetSSandDD()
		ds_train.add_from_dataset(self.extensions)
		ds_test.add_from_dataset(self.originals)
		if shuffle:
			ds_test.shuffle(olds=olds)
		ds_leftover = ds_test.truncate(ratio)
		ds_train.add_from_dataset([ds_leftover])
		if shuffle:
			ds_train.shuffle()
		return ds_train, ds_test

	def split_by_index(self, train_index, test_index, shuffle=False):
		if self.task == 'ss':
			ds_train = DataSetSS()
			ds_test = DataSetSS()
		elif self.task == 'dd_fillers':
			ds_train = DataSetFillers()
			ds_test = DataSetFillers()
		elif self.task == 'dd_editdisfs':
			ds_train = DataSetEditDisfs()
			ds_test = DataSetEditDisfs()
		elif self.task == 'dd_editdisfs_binary':
			ds_train = DataSetEditDisfs(binary=True)
			ds_test = DataSetEditDisfs(binary=True)
		else:
			ds_train = DataSetSSandDD()
			ds_test = DataSetSSandDD()
		ds_train.add_from_dataset(self.originals, index=train_index)
		ds_train.add_from_dataset(self.extensions)
		ds_test.add_from_dataset(self.originals, index=test_index)
		if shuffle:
			ds_test.shuffle()
			ds_train.shuffle()
		return ds_train, ds_test


class DataSetSS:

	def __init__(self, binary=False):
		self.texts = []
		self.word_texts = []
		self.word_id_texts = []
		self.pros_texts = []
		self.shuffle_indexes = []
		self.binary = binary
		tokenizer = None

	def indexes_to_words(self, indexes_texts):
		return tokenizer.index_texts_to_text(indexes_texts)

	def add_from_dataset(self, datasets=[], index=None):
		for ds in datasets:
			self.texts.extend(ds.texts)
			self.word_texts.extend(ds.word_texts)
			self.word_id_texts.extend(ds.word_id_texts)
			self.pros_texts.extend(ds.pros_texts)
		self.shuffle_indexes = list(range(len(self.word_texts)))
		if index is not None:
			self.texts = [self.texts[i] for i in index]
			self.word_texts = [self.word_texts[i] for i in index]
			self.word_id_texts = [self.word_id_texts[i] for i in index]
			self.pros_texts = [self.pros_texts[i] for i in index]
			self.shuffle_indexes = [i for i in index]

	def as_matrix(self, ids=True):
		x, y = [], []
		end_period = tokenizer.word_index['.'] if ids else '.'
		texts = self.word_id_texts if ids else self.word_texts
		for i, text in enumerate(texts):
			x_, y_ = [], []
			for word in text:
				if word == end_period:
					y_[-1] = 1
				else:
					x_.append(word)
					y_.append(0)
			y_[-1] = 1
			x.append(x_)
			y.append(y_)
		return x, y

	def shuffle(self, olds=[]):
		shuffle_list(self.shuffle_indexes)
		t, wt, wi, pt, si = [], [], [], [], []
		for i in self.shuffle_indexes:
			if i not in olds:
				t.append(self.texts[i])
				wt.append(self.word_texts[i])
				wi.append(self.word_id_texts[i])
				pt.append(self.pros_texts[i])
				si.append(i)
		for i in self.shuffle_indexes:
			if i in olds:
				t.append(self.texts[i])
				wt.append(self.word_texts[i])
				wi.append(self.word_id_texts[i])
				pt.append(self.pros_texts[i])
				si.append(i)
		self.texts = t
		self.word_texts = wt
		self.word_id_texts = wi
		self.pros_texts = pt
		self.shuffle_indexes = si

	def truncate(self, ratio):
		limit = len(self.texts) - int(len(self.texts) * ratio)
		ds_leftover = DataSetSS()
		ds_leftover.texts = self.texts[limit:]
		ds_leftover.word_texts = self.word_texts[limit:]
		ds_leftover.word_id_texts = self.word_id_texts[limit:]
		ds_leftover.pros_texts = self.pros_texts[limit:]
		ds_leftover.shuffle_indexes = self.shuffle_indexes[limit:]
		self.texts = self.texts[:limit]
		self.word_texts = self.word_texts[:limit]
		self.word_id_texts = self.word_id_texts[:limit]
		self.pros_texts = self.pros_texts[:limit]
		self.shuffle_indexes = self.shuffle_indexes[:limit]
		return ds_leftover

	def info(self):
		nb_texts = len(self.texts)
		nb_sentences = sum(map(lambda x: x.count('.'), self.texts))
		nb_words = sum(map(len, self.word_id_texts))
		logger.info(type(self).__name__)
		logger.info('Nb texts: {}'.format(nb_texts))
		logger.info('Nb sentences: {}'.format(nb_sentences))
		logger.info('Nb words: {}'.format(nb_words))
		logger.info('Chance Baseline: {0:.4%}'.format(nb_sentences/(nb_sentences + nb_words + int(nb_words==0))))


class DataSetFillers(DataSetSS):

	# 'Pausas-preenchidas': 		'*'
	# 'Marcadores-discursivos': 	'+'
	# 'Termo-de-ed-exp': 			'$'
	def as_matrix(self, ids=True):
		x, y = [], []
		end_period 	= tokenizer.word_index['.'] if ids else '.'
		end_pp 		= tokenizer.word_index['*'] if ids else '*'
		end_md 		= tokenizer.word_index['+'] if ids else '+'
		end_tee 	= tokenizer.word_index['$'] if ids else '$'
		texts = self.word_id_texts if ids else self.word_texts
		for i, text in enumerate(texts):
			x_, y_ = [], []
			for word in text:
				if word == end_pp:
					y_[-1] = 1
				elif word == end_md:
					y_[-1] = 2
				elif word == end_tee:
					# y_[-1] = 3
					# muito pouco tee
					pass
				elif word == end_period:
					pass
				else:
					x_.append(word)
					y_.append(0)
			x.append(x_)
			y.append(y_)
		return x, y

	def info(self):
		nb_texts = len(self.texts)
		nb_sentences = sum(map(lambda x: x.count('.'), self.texts))
		nb_pps = sum(map(lambda x: x.count('*'), self.texts))
		nb_mds = sum(map(lambda x: x.count('+'), self.texts))
		nb_tees = sum(map(lambda x: x.count('$'), self.texts))
		nb_words = sum(map(len, self.word_id_texts))
		logger.info(type(self).__name__)
		logger.info('Nb texts: {}'.format(nb_texts))
		logger.info('Nb sentences: {}'.format(nb_sentences))
		logger.info('Nb pps: {}'.format(nb_pps))
		logger.info('Nb mds: {}'.format(nb_mds))
		logger.info('Nb tees: {}'.format(nb_tees))
		logger.info('Nb words: {}'.format(nb_words))
		logger.info('Chance Baseline: {0:.4%}'.format(nb_sentences/(nb_sentences + nb_words + int(nb_words==0))))


class DataSetEditDisfs(DataSetSS):

	# ANN_SCHEME = {
	# 	'EO-Repeticao': '*',
	# 	'EO-Revisao': 	'+',
	# 	'EO-Recomeco': 	'$'
	# }

	# BIES_SCHEME = {
	# 	'IP': 		'|',
	# 	'BEGIN': 	'<',
	# 	'MIDDLE': 	'=',
	# 	'END': 		'>',
	# 	'SINGLE': 	'#'
	# }
	def as_matrix(self, ids=True):
		x, y = [], []
		if self.binary:
			end_period 	= tokenizer.word_index['.'] if ids else '.'
			end_dd 	= tokenizer.word_index['*'] if ids else '*'
			texts = self.word_id_texts if ids else self.word_texts
			for i, text in enumerate(texts):
				x_, y_ = [], []
				for word in text:
					if word == end_dd:
						y_[-1] = 1
					elif word == end_period:
						pass
					else:
						x_.append(word)
						y_.append(0)
				x.append(x_)
				y.append(y_)
		else:
			end_period 	= tokenizer.word_index['.'] if ids else '.'
			end_rep 	= tokenizer.word_index['*'] if ids else '*'
			end_rev 	= tokenizer.word_index['+'] if ids else '+'
			end_rec 	= tokenizer.word_index['$'] if ids else '$'
			texts = self.word_id_texts if ids else self.word_texts
			for i, text in enumerate(texts):
				x_, y_ = [], []
				for word in text:
					if word == end_rep:
						y_[-1] = 1
					elif word == end_rev:
						y_[-1] = 2
					elif word == end_rec:
						y_[-1] = 3
					elif word == end_period:
						pass
					else:
						x_.append(word)
						y_.append(0)
				x.append(x_)
				y.append(y_)
		return x, y

	def info(self):
		nb_texts = len(self.texts)
		nb_sentences = sum(map(lambda x: x.count('.'), self.texts))
		nb_reps = sum(map(lambda x: x.count('*'), self.texts))
		nb_revs = sum(map(lambda x: x.count('+'), self.texts))
		nb_recs = sum(map(lambda x: x.count('$'), self.texts))
		nb_words = sum(map(len, self.word_id_texts))
		logger.info(type(self).__name__)
		logger.info('Nb texts: {}'.format(nb_texts))
		logger.info('Nb sentences: {}'.format(nb_sentences))
		logger.info('Nb reps: {}'.format(nb_reps))
		logger.info('Nb revs: {}'.format(nb_revs))
		logger.info('Nb recs: {}'.format(nb_recs))
		logger.info('Nb words: {}'.format(nb_words))
		logger.info('Chance Baseline: {0:.4%}'.format(nb_sentences/(nb_sentences + nb_words + int(nb_words==0))))


class DataSetSSandDD(DataSetSS):

	def as_matrix(self, ids=True):
		x, y = [], []
		end_period = tokenizer.word_index['.'] if ids else '.'
		end_disf = tokenizer.word_index['*'] if ids else '*'
		texts = self.word_id_texts if ids else self.word_texts
		for i, text in enumerate(texts):
			x_, y_ = [], []
			for word in text:
				if word == end_period:
					if y_[-1] == 2:
						y_[-1] = 3
					else:
						y_[-1] = 1
				elif word == end_disf:
					y_[-1] = 2
				else:
					x_.append(word)
					y_.append(0)
			y_[-1] = 1
			x.append(x_)
			y.append(y_)
		return x, y

