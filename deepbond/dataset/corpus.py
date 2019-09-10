import logging

from torchtext import data
from deeptagger.dataset.cleaner import Cleaner
from deeptagger.features import (extract_prefixes, extract_suffixes,
                                 extract_caps)


class Corpus:

    def __init__(self, fields_tuples, delimiter_word=' ', delimiter_tag='/'):
        """
        Base class for a PoS Corpus.
        :param fields_tuples: a list of tuples where the first element is an
                              attr  name and the second is a torchtext's Field
                              object.
        :param delimiter_word: char that delimiters tokens
        :param delimiter_tag: har that delimiters word tokens from tags tokens
        """
        # list of fields containing the same number of examples
        self.fields_examples = []
        # list of name of attrs and their corresponding torchtext fields
        self.attr_fields = fields_tuples
        # delimiters
        self.del_word = delimiter_word
        self.del_tag = delimiter_tag
        # the number of examples in the corpus
        self.nb_examples = 0
        # mapping from attr name to their index in the list
        names, _ = zip(*self.attr_fields)
        self.attr_index = dict(zip(names, range(len(names))))
        self.fields_examples = [[] for _ in range(len(self.attr_index))]

    def add_texts(self, texts):
        """
        Add a list of texts to the corpus.
        :param texts: just a string where tokens are separated by a single
                      space or a list of strings to add more than one text
        """
        if not isinstance(texts, (list, tuple)):
            texts = [texts]
        texts = list(map(self._normalize, texts))
        self.fields_examples[self.attr_index['words']] = texts
        self.nb_examples += len(texts)

    def add_features(self,
                     features_fields_tuples,
                     prefix_min_length=1,
                     prefix_max_length=5,
                     suffix_min_length=1,
                     suffix_max_length=5):
        """
        Add features in features_field_tuples. Accepted features are:
        prefixes, suffixes and caps
        :param features_fields_tuples: the same as fields_tuples but it
                                       represents optional features

        :param prefix_min_length: min length for prefixes
        :param prefix_max_length: max length for prefixes
        :param suffix_min_length: min length for suffixes
        :param suffix_max_length: max length for suffixes
        :return:
        """
        new_examples = [[] for _ in range(len(features_fields_tuples))]
        self.fields_examples.extend(new_examples)
        words = self.fields_examples[self.attr_index['words']]
        if 'prefixes' in self.attr_index:
            prefixes = extract_prefixes(words,
                                        prefix_min_length,
                                        prefix_max_length)
            self.fields_examples[self.attr_index['prefixes']] = prefixes
        if 'suffixes' in self.attr_index:
            suffixes = extract_suffixes(words,
                                        suffix_min_length,
                                        suffix_max_length)
            self.fields_examples[self.attr_index['suffixes']] = suffixes
        if 'caps' in self.attr_index:
            self.fields_examples[self.attr_index['caps']] = extract_caps(words)

    def read(self, filepath):
        """
        filepath: path to a file with the following format:
                  words are delimited by `delimiter_word` and
                  tags are delimited from words by `delimiter_tag`
                  e.g. The_ART princess_S is_V pretty_ADJ
                  where delimiter_word=' ' and delimiter_tag='_'
        """
        # warning for two well known Brazilian Portuguese PoS corpus
        if 'macmorpho' in filepath and self.del_tag != '_':
            logging.warning('Default MacMorpho delimiter tag is `_`, '
                            'but you passed `{}`'.format(self.del_tag))
        if 'tychobrahe' in filepath and self.del_tag != '/':
            logging.warning('Default TychoBrahe delimiter tag is `/`, '
                            'but you passed `{}`'.format(self.del_tag))

        # load the file and fill words and tags examples
        words_for_example = []
        tags_for_example = []
        with open(filepath, 'r', encoding='utf8') as f:
            for line in f:
                line = Cleaner.trim(line.strip())
                words, tags = zip(
                    *list(
                        map(
                            lambda x: x.rsplit(self.del_tag, 1),
                            line.split(self.del_word)
                        )
                    )
                )
                words_for_example.append(self._normalize(' '.join(words)))
                tags_for_example.append(' '.join(tags))
        # add words and tags examples
        self.fields_examples[self.attr_index['words']] = words_for_example
        if 'tags' in self.attr_index:
            self.fields_examples[self.attr_index['tags']] = tags_for_example
        # then add each corresponding sentence from each field
        nb_lines = [len(fe) for fe in self.fields_examples if fe]
        # assert files have the same size
        assert min(nb_lines) == max(nb_lines)
        self.nb_examples = nb_lines[0]

    def __iter__(self):
        for j in range(self.nb_examples):
            fields_values_for_example = [self.fields_examples[i][j]
                                         for i in range(len(self.attr_fields))]
            yield data.Example.fromlist(fields_values_for_example,
                                        self.attr_fields)

    @staticmethod
    def _normalize(text):
        t = text.strip()
        t = Cleaner.trim(t)
        t = Cleaner.transform_numbers(t)
        # t = Cleaner.transform_decimals(t)
        # t = Cleaner.transform_urls(t)
        # t = Cleaner.transform_dollar(t)
        # t = Cleaner.transform_dates(t)
        # t = Cleaner.transform_hours(t)
        # t = Cleaner.transform_emails(t)
        # t = Cleaner.fix_quotes(t)
        # t = Cleaner.fix_mistyped_tokens(t)
        return t
