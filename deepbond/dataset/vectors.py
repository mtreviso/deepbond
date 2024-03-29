import logging
from functools import partial

import torch
from torchtext.vocab import Vectors

from deepbond.constants import UNK, PAD, START, STOP

logger = logging.getLogger(__name__)


class WordEmbeddings(Vectors):
    def __init__(
        self,
        name,
        emb_format='polyglot',
        binary=True,
        map_fn=lambda x: x,
        **kwargs
    ):
        """
        Arguments:
           emb_format: the saved embedding model format, choices are:
                       polyglot, word2vec, fasttext, glove and text
           binary: only for word2vec, fasttext and text
           map_fn: a function that maps special original tokens
                       to Polyglot tokens (e.g. <eos> to </S>)
           save_vectors: save a vectors cache
        """
        self.binary = binary
        self.emb_format = emb_format

        self.itos = None
        self.stoi = None
        self.dim = None
        self.vectors = None
        self.unk_vector = None

        self.map_fn = map_fn
        super().__init__(name, **kwargs)

    def __getitem__(self, token):
        mapped_token = self.map_fn(token)
        if self.emb_format == 'fasttext':
            if mapped_token in self.vectors:
                return torch.from_numpy(self.vectors[mapped_token])
            elif mapped_token.lower() in self.vectors:
                print(mapped_token)
                return torch.from_numpy(self.vectors[mapped_token.lower()])
            else:
                print(mapped_token)
                return self.unk_vector.clone()

        if mapped_token in self.stoi:
            return self.vectors[self.stoi[mapped_token]]
        elif token in self.stoi:
            return self.vectors[self.stoi[token]]
        elif self.unk_vector is not None:
            return self.unk_vector.clone()
        else:
            return self.unk_init(torch.Tensor(1, self.dim))

    def cache(self, name, cache, url=None, max_vectors=None):
        if self.emb_format == 'polyglot':
            try:
                from polyglot.mapping import Embedding
            except ImportError:
                logger.error('Please install `polyglot` package first.')
                return None
            embeddings = Embedding.load(name)
            self.itos = embeddings.vocabulary.id_word
            self.stoi = embeddings.vocabulary.word_id
            self.dim = embeddings.shape[1]
            self.vectors = torch.tensor(embeddings.vectors).view(-1, self.dim)

        elif self.emb_format == 'glove':
            itos = []
            vectors = []
            with open(name, 'r', encoding='utf8') as f:
                for line in f:
                    try:
                        values = line.rstrip().split()
                        itos.append(values[0])
                        vectors.append([float(x) for x in values[1:]])
                    except ValueError:
                        # ignore entries that look like:
                        # by name@domain.com 0.6882 -0.36436 ...
                        continue
            self.itos = itos
            self.stoi = dict(zip(self.itos, range(len(self.itos))))
            self.dim = len(vectors[0])
            self.vectors = torch.tensor(vectors).view(-1, self.dim)

        elif self.emb_format == 'fasttext':
            try:
                from gensim.models import FastText
            except ImportError:
                logger.error('Please install `gensim` package first.')
                return None
            self.vectors = FastText.load_fasttext_format(name)
            self.itos = list(self.vectors.wv.vocab.keys())
            self.stoi = dict(zip(self.itos, range(len(self.itos))))
            self.unk_vector = self.vectors['<unk>']
            self.dim = self.vectors.vector_size

        elif self.emb_format == 'word2vec':
            try:
                from gensim.models import KeyedVectors
            except ImportError:
                logger.error('Please install `gensim` package first.')
                return None
            embeddings = KeyedVectors.load_word2vec_format(
                name, unicode_errors='ignore', binary=self.binary
            )
            self.itos = embeddings.index2word
            self.stoi = dict(zip(self.itos, range(len(self.itos))))
            self.dim = embeddings.vector_size
            self.vectors = torch.tensor(embeddings.vectors).view(-1, self.dim)

        elif self.emb_format == 'text':
            tokens = []
            vectors = []
            if self.binary:
                import pickle

                # vectors should be a dict mapping str keys to numpy arrays
                with open(name, 'rb') as f:
                    d = pickle.load(f)
                    tokens = list(d.keys())
                    vectors = list(d.values())
            else:
                # each line should contain a token and its following fields
                # <token> <vector_value_1> ... <vector_value_n>
                with open(name, 'r', encoding='utf8') as f:
                    for line in f:
                        if line:  # ignore empty lines
                            fields = line.rstrip().split()
                            tokens.append(fields[0])
                            vectors.append(list(map(float, fields[1:])))
            self.itos = tokens
            self.stoi = dict(zip(self.itos, range(len(self.itos))))
            self.vectors = torch.tensor(vectors)
            self.dim = self.vectors.shape[1]

        elif self.emb_format == 'fonseca':
            import numpy as np
            import os
            embeddings = np.load(os.path.join(name, 'types-features.npy'))
            texts = open(os.path.join(name, 'vocabulary.txt'), 'r', encoding='utf8').read()
            words = set([w.strip() for w in texts.split('\n')])
            self.itos = list(words)
            self.stoi = dict(zip(self.itos, range(len(self.itos))))
            self.dim = embeddings.shape[1]
            self.vectors = torch.tensor(embeddings).view(-1, self.dim)

        if self.unk_vector is None:
            self.unk_vector = self.vectors.mean(0).unsqueeze(0)


def to_polyglot(token):
    mapping = {
        UNK: '<UNK>',
        PAD: '<PAD>',
        START: '<S>',
        STOP: '</S>'
    }
    if token in mapping:
        return mapping[token]
    return token


def to_fonseca(token):
    mapping = {
        UNK: '*rare*',
        PAD: '*right*',
        START: '*left*',
        STOP: '*right*'
    }
    if token in mapping:
        return mapping[token]
    return token


Polyglot = partial(WordEmbeddings, emb_format='polyglot', map_fn=to_polyglot)
Word2Vec = partial(WordEmbeddings, emb_format='word2vec')
FastText = partial(WordEmbeddings, emb_format='fasttext')
Glove = partial(WordEmbeddings, emb_format='glove')
Fonseca = partial(WordEmbeddings, emb_format='fonseca', map_fn=to_fonseca)
TextVectors = partial(WordEmbeddings, emb_format='text')

available_embeddings = {
    'polyglot': Polyglot,
    'word2vec': Word2Vec,
    'fasttext': FastText,
    'glove': Glove,
    'fonseca': Fonseca,
    'text': TextVectors
}
