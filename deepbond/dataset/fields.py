import logging
from collections import defaultdict
from pathlib import Path

import torch
from torchtext.data import Field

from deepbond import constants
from deepbond.dataset.vocabulary import Vocabulary
from deepbond.dataset.vectors import (Polyglot,
                                      Word2Vec,
                                      FastText,
                                      Glove,
                                      Fonseca)


available_embeddings = {
    'polyglot': Polyglot,
    'word2vec': Word2Vec,
    'fasttext': FastText,
    'glove': Glove,
    'fonseca': Fonseca,
}


def load_vectors(options):
    vectors = None
    if options.embeddings_format is not None:
        # load the word embeddings only if a correct format is provided
        assert options.embeddings_format in available_embeddings.keys()

        logging.info('Loading {} word embeddings from: {}'.format(
            options.embeddings_format, options.embeddings_path)
        )
        word_emb_cls = available_embeddings[options.embeddings_format]
        vectors = word_emb_cls(options.embeddings_path, binary=False)
    return vectors


def build_vocabs(fields_tuples, train_dataset, all_datasets, options):
    # load word embeddings
    vectors = load_vectors(options)

    # transform fields_tuples to a dict in order to access fields easily
    dict_fields = defaultdict(lambda: None)
    dict_fields.update(dict(fields_tuples))
    words_field = dict_fields['words']
    tags_field = dict_fields['tags']

    # build vocab for words based on the training set
    words_field.build_vocab(
        train_dataset,
        vectors=vectors,
        max_size=options.vocab_size,
        min_freq=options.vocab_min_frequency,
        keep_rare_with_vectors=options.keep_rare_with_vectors,
        add_vectors_vocab=options.add_embeddings_vocab
    )

    # build vocab based on all datasets
    tags_field.build_vocab(*all_datasets, specials_first=False)

    # set global constants to their correct value
    constants.PAD_ID = dict_fields['words'].vocab.stoi[constants.PAD]
    constants.TAGS_PAD_ID = dict_fields['tags'].vocab.stoi[constants.PAD]


def load_vocabs(path, fields_tuples):
    vocab_path = Path(path, constants.VOCAB)

    # load vocabs for each field and transform it to dict to access it easily
    vocabs = torch.load(str(vocab_path),
                        map_location=lambda storage, loc: storage)
    vocabs = dict(vocabs)

    # set field.vocab to its correct vocab object
    for name, field in fields_tuples:
        field.vocab = vocabs[name]

    # transform fields_tuples to a dict in order to access fields easily
    dict_fields = dict(fields_tuples)

    # ensure global constants to their correct value
    constants.PAD_ID = dict_fields['words'].vocab.stoi[constants.PAD]
    constants.TAGS_PAD_ID = dict_fields['tags'].vocab.stoi[constants.PAD]


def save_vocabs(path, fields_tuples):
    # list of fields name and their vocab
    vocabs = []
    for name, field in fields_tuples:
        vocabs.append((name, field.vocab))

    # save vectors in a temporary dict and save the vocabs
    vectors = {}
    for name, vocab in vocabs:
        vectors[name] = vocab.vectors
        vocab.vectors = None
    vocab_path = Path(path, constants.VOCAB)
    torch.save(vocabs, str(vocab_path))

    # restore vectors -> useful if we want to use fields later
    for name, vocab in vocabs:
        vocab.vectors = vectors[name]


class WordsField(Field):
    """Defines a field for word tokens with default
       values from constant.py and with the vocabulary
       defined in vocabulary.py."""

    def __init__(self, **kwargs):
        super().__init__(unk_token=constants.UNK,
                         pad_token=constants.PAD,
                         init_token=constants.START,
                         eos_token=constants.STOP,
                         batch_first=True,
                         **kwargs)
        self.vocab_cls = Vocabulary


class TagsField(Field):
    """Defines a field for tag tokens by setting unk_token to None
       and pad_token to constants.PAD as default."""

    def __init__(self, **kwargs):
        super().__init__(unk_token=None,
                         pad_token=constants.PAD,
                         is_target=True,
                         batch_first=True,
                         **kwargs)
        self.vocab_cls = Vocabulary
