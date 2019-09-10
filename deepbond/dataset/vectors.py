import logging
from functools import partial

import torch
from torchtext.vocab import Vectors

from deeptagger.constants import UNK, PAD, START, STOP


class WordEmbeddings(Vectors):

    def __init__(self, name, emb_format='polyglot', binary=False,
                 map_fn=lambda x: x, save_vectors=False, **kwargs):
        """
        Arguments:
           emb_format: the saved embedding model format, choices are:
                       polyglot, word2vec, fasttext and glove
           binary: wheter the saved emb file is in binary
           map_fn: a function that maps special original tokens
                       to Polyglot tokens (e.g. <eos> to </S>)
           save_vectors: save a vectors cache
        """
        self.itos = []
        self.stoi = {}
        self.dim = None
        self.vectors = None
        self.binary = binary
        self.emb_format = emb_format
        self.map_fn = map_fn
        self.save_vectors = save_vectors
        self.unk_vector = None
        super().__init__(name, **kwargs)

    def __getitem__(self, token):
        mapped_token = self.map_fn(token)
        if mapped_token in self.stoi:
            return self.vectors[self.stoi[mapped_token]]
        elif token in self.stoi:
            return self.vectors[self.stoi[token]]
        elif self.unk_vector is not None:
            return self.unk_vector.clone()
        else:
            return self.unk_init(torch.Tensor(1, self.dim))

    def cache(self, name, cache, url=None, max_vectors=None):
        if self.emb_format in ['polyglot', 'glove']:
            from polyglot.mapping import Embedding
            if self.emb_format == 'polyglot':
                embeddings = Embedding.load(name)
            else:
                embeddings = Embedding.from_glove(name)
            self.itos = embeddings.vocabulary.id_word
            self.stoi = embeddings.vocabulary.word_id
            self.dim = embeddings.shape[1]
            self.vectors = torch.Tensor(embeddings.vectors).view(-1, self.dim)
        elif self.emb_format in ['word2vec', 'fasttext']:
            try:
                from gensim.models import KeyedVectors
            except ImportError:
                logging.error('Please install `gensim` package first.')

            embeddings = KeyedVectors.load_word2vec_format(
                name, unicode_errors='ignore', binary=self.binary
            )
            self.itos = embeddings.index2word
            self.stoi = dict(zip(self.itos, range(len(self.itos))))
            self.dim = embeddings.vector_size
            self.vectors = torch.Tensor(embeddings.vectors).view(-1, self.dim)
        elif self.emb_format == 'fonseca':
            import numpy as np
            import os
            embeddings = np.load(os.path.join(name, 'types-features.npy'))
            texts = open(os.path.join(name, 'vocabulary.txt'), 'r').read()
            words = set([w.strip() for w in texts.split('\n')])
            self.itos = list(words)
            self.stoi = dict(zip(self.itos, range(len(self.itos))))
            self.dim = embeddings.shape[1]
            self.vectors = torch.Tensor(embeddings).view(-1, self.dim)
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
