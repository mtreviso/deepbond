import re
from pathlib import Path

from torchtext import data


class Corpus:

    word_label = '_'

    def __init__(self, fields_tuples, punctuations='!#$%&*+,-./:;<=>?@^|~'):
        """
        Base class for a SS/DD Corpus.

        Args:
            fields_tuples (list of str, torchtext.field): a list of tuples
                where the first element is an attr  name and the second is
                a torchtext's Field object.
            punctuations (str): the tagset to be considered. e.g. for '.?!'
                three punctuations are considered as tags: , ? and !
        """
        # assert we have some punctuations and they are non-alpha chars
        assert len(punctuations) > 0
        for c in list(punctuations):
            assert not c.isalpha()

        # list of fields containing the same number of examples
        self.fields_examples = []
        # list of name of attrs and their corresponding torchtext fields
        self.attr_fields = fields_tuples
        # punctuation to be consi
        self.punctuations = punctuations
        # mapping from attr name to their index in the list
        names, _ = zip(*self.attr_fields)
        self.fields_examples = {k: [] for k in names}

    @property
    def nb_examples(self):
        return len(self.fields_examples['words'])

    def read(self, corpus_path):
        """
        Args:
            path_to (str): path to dir where files have the following format:
                  words are delimited by spaces and tags are punctuation marks
                  Everything else is treated as text.
                  Or path to a file where each instance is delimited by a
                  new line. The format for each instance is the same to
                  the dir case.
        """

        # load the file and fill words and tags examples
        corpus_path = Path(corpus_path)
        if corpus_path.is_dir():
            words_for_example, tags_for_example = self.read_dir(corpus_path)
        else:
            words_for_example, tags_for_example = self.read_file(corpus_path)

        # add words and tags examples
        self.fields_examples['words'] = words_for_example
        self.fields_examples['tags'] = tags_for_example

        # assert files have the same size
        nb_lines = [len(fe) for fe in self.fields_examples.values() if fe]
        assert min(nb_lines) == max(nb_lines)

        # assert that the number of words are equal to the number of tags for
        # each file
        for words, tags in zip(self.fields_examples['words'],
                               self.fields_examples['tags']):
            assert len(words.split()) == len(tags.split())

    def read_dir(self, dir_path):
        words_for_example = []
        tags_for_example = []
        for f_path in sorted(dir_path.iterdir()):
            with f_path.open('r', encoding='utf8') as f:
                paragraph = self._normalize(f.read())
                words, tags = self.get_words_and_tags(paragraph)
                words_for_example.append(' '.join(words))
                tags_for_example.append(' '.join(tags))
        return words_for_example, tags_for_example

    def read_file(self, file_path):
        words_for_example = []
        tags_for_example = []
        with file_path.open('r', encoding='utf8') as f:
            for line in f:
                paragraph = self._normalize(line)
                words, tags = self.get_words_and_tags(paragraph)
                words_for_example.append(' '.join(words))
                tags_for_example.append(' '.join(tags))
        return words_for_example, tags_for_example

    def get_words_and_tags(self, text):
        """
        Extract word tokens and labels for a given text.
        Words are non-punctuation symbols. Labels are the punctuations
        themselves and a dummy-label for words.

        Args:
            text (str): a stream of chars

        Returns:
            list of str (words): list of non-punctuation tokens
            list of str (tags): list with punctuations and fill-label for words
        """
        words = []
        tags = []
        for token in text.split():
            if token in self.punctuations:
                if len(tags) == 0:
                    tags.append(token)
                else:
                    tags[-1] = token
            else:
                words.append(token)
                tags.append(self.word_label)
        return words, tags

    def __iter__(self):
        for j in range(self.nb_examples):
            fields_for_example = [
                self.fields_examples[k][j] for k in self.fields_examples.keys()
            ]
            yield data.Example.fromlist(fields_for_example, self.attr_fields)

    @staticmethod
    def _normalize(text):
        """
        Put a space between . , ? ! : ; * and words
        Args:
            text (str): stream of words
        """
        t = text.strip()
        t = re.sub(r'(\S+)([\.\,\?\!\:\;\*])', r'\1 \2', t.strip())
        t = re.sub(r'([\.\,\?\!\:\;\*])(\S+)', r'\1 \2', t.strip())
        t = re.sub(r'\ +', ' ', t.strip())
        return t

    def add_texts(self, texts):
        """
        Add a list of texts to the corpus.

        Args:
            texts (str): just a string where tokens are separated by a single
                         space or a list of strings to add more than one text
        """
        if not isinstance(texts, (list, tuple)):
            texts = [texts]

        words_for_example = []
        for text in texts:
            text = self._normalize(text)
            words, _ = self.get_words_and_tags(text)
            words_for_example.append(' '.join(words))

        self.fields_examples['words'] = words_for_example
