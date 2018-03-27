import pickle 
from deepbond.helpers import ProbabilisticTagger
from deepbond.utils import unroll

class POS:

	def __init__(self, type='nlpnet'):
		self.type = type

	def load(self, filename):
		if self.type == 'nlpnet':
			import nlpnet
			self.pos_tagger = nlpnet.POSTagger(filename)
			self.vocabulary = dict(zip(self.pos_tagger.itd.values(), range(1, len(self.pos_tagger.itd)+1)))
		else:
			# self._save_POS(filename)
			self.pos_tagger = pickle.load(open(filename, 'rb'))
			self.vocabulary = self.pos_tagger.vocabulary()

	def _save_POS(self, filename):
		self.pos_tagger = ProbabilisticTagger(uppercase=False)
		with open(filename, 'wb') as handle:
			pickle.dump(self.pos_tagger, handle)

	def get(self, texts):
		if self.type == 'nlpnet':
			tags = [[x[1] for x in unroll(self.pos_tagger.tag(' '.join(text)))] for text in texts]
		else:
			tags = [[x[1] for x in self.pos_tagger.tag(text)] for text in texts]
		ids = set([i for t in tags for i in t])
		if self.vocabulary is None:
			self.vocabulary = dict(zip(ids, range(1, len(ids)+1)))
		return [list(map(self.vocabulary.__getitem__, t)) for t in tags]