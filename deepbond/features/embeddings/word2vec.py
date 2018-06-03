import logging
import gensim
from .embedding import Embedding

class Word2Vec(Embedding):

	@property
	def vocabulary(self): 
		if self._vocabulary is None:
			self._vocabulary = dict(zip(self.model.vocab.keys(), range(len(self.model.vocab))))
		return self._vocabulary

	def load(self, load_file):
		self.model = gensim.models.Word2Vec.load(load_file)
		self.dimensions = self.model.vector_size

	def save(self, save_file):
		self.model.save(save_file)

	def train(self, train_file, **kwargs):
		logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
		self.model = gensim.models.Word2Vec(gensim.models.word2vec.LineSentence(train_file),
											size=self.dimensions, window=self.window_size, iter=self.epochs,
											min_count=self.min_count, workers=self.workers, **kwargs)