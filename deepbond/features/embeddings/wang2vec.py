from .embedding import LoadableEmbedding

class Wang2Vec(LoadableEmbedding):
	"""
	A class that just load the vectors in specified file and
	return a np.array with shape (vocabulary_size, word_embedding_dimension)
	see: https://github.com/wlin12/wang2vec
	TODO: wrapper c train/save
	---
	Attention: the output_embeddings should be in plain text (not binary)
	"""
	def load(self, load_file):
		self.model = dict()
		ignore_first_line = True # metadata
		vocab_size, word_dim = 0, 0
		with open(load_file, 'r', encoding="utf-8") as f:
			for line in f:
				if ignore_first_line:
					vocab_size, word_dim = map(int, line.split())
					ignore_first_line = False
					continue
				data = line.split()
				self.model[data[0]] = list(map(float, data[1:]))
		self.dimensions = word_dim