from collections import Counter, OrderedDict
from nltk.tokenize import RegexpTokenizer

# works in Python 2 & 3
class _Singleton(type):
	""" A metaclass that creates a Singleton base class when called. """
	_instances = {}
	def __call__(cls, *args, **kwargs):
		if cls not in cls._instances:
			cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
		return cls._instances[cls]

class Singleton(_Singleton('SingletonMeta', (object,), {})): 
	pass


class SentTokenizer(Singleton):

	def __init__(self):
		self.unknown_word = '*rare*'
		self.unknown_word_id = 0
		self.word_count = OrderedDict({'.': 0}) # add '.' to the vocabulary
		self.word_index = OrderedDict()
		self.index_word = OrderedDict()
		self.labels = [' ', '.']
		self.tokenizer = RegexpTokenizer(r'\S+') #\w+|\$[\d\.]+|\S+
		self.loaded = False

	def reset(self):
		self.unknown_word = '*rare*'
		self.unknown_word_id = 0
		self.word_count = OrderedDict({'.': 0}) # add '.' to the vocabulary
		self.word_index = OrderedDict()
		self.index_word = OrderedDict()
		self.labels = [' ', '.']
		self.tokenizer = RegexpTokenizer(r'\S+') #\w+|\$[\d\.]+|\S+
		self.loaded = False		

	def load_vocabulary(self, filename):
		with open(filename, 'r') as f:
			for line in f:
				if line.strip():
					word = line.strip()
					self.word_count[word] = 0
		self.loaded = True
		self.build_vocab()

	def save_vocabulary(self, filename):
		f = open(filename, 'w', encoding='utf8')
		for word in self.word_index.keys():
			f.write('{}\n'.format(word))
		f.close()

	def fit_on_texts(self, texts, build_vocab=True):
		word_texts = []
		for text in texts:
			if text.strip():
				word_sequence = self.text_to_word_sequence(text)
				word_texts.append(word_sequence)
				self._count_words(word_sequence)
		if build_vocab:
			self.build_vocab()
		return word_texts

	def text_to_word_sequence(self, text, normalize=False):
		if normalize:
			return self.tokenizer.tokenize(self._normalize(text))
		return self.tokenizer.tokenize(text)

	def _normalize(self, text):
		return text

	def _count_words(self, word_sequence):
		for word in word_sequence:
			if word not in self.word_count:
				self.word_count[word] = 0
			self.word_count[word] += 1
	
	def build_vocab(self):
		words_keys = list(self.word_count.keys())
		words_indexes = list(range(self.unknown_word_id+1, len(words_keys)+self.unknown_word_id+1))
		self.word_index = OrderedDict(zip(words_keys, words_indexes))
		self.index_word = OrderedDict(zip(self.word_index.values(), self.word_index.keys()))

	def text_to_indexes(self, word_sequence):
		return [self.word_index.get(x) if x in self.word_index else self.unknown_word_id for x in word_sequence]

	def word_texts_to_indexes(self, texts):
		return [self.text_to_indexes(t) for t in texts]

	def indexes_to_text(self, index_sequence):
		return [self.index_word.get(x) if x in self.index_word else self.unknown_word for x in index_sequence]

	def index_texts_to_text(self, texts):
		return [self.indexes_to_text(t) for t in texts]

