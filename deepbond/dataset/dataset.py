from torchtext.data import Dataset

from deeptagger.dataset.corpus import Corpus


def build(path, fields_tuples, options):
    def filter_len(x):
        return options.min_length <= len(x.words) <= options.max_length
    corpus = Corpus(fields_tuples, options.del_word, options.del_tag)
    corpus.read(path)
    feature_fields = list(filter(lambda x: x[0] not in ['words', 'tags'],
                                 fields_tuples))
    if feature_fields:
        corpus.add_features(feature_fields,
                            options.prefix_min_length,
                            options.prefix_max_length,
                            options.suffix_min_length,
                            options.suffix_max_length)
    return PoSDataset(corpus, filter_pred=filter_len)


def build_texts(texts, fields_tuples, options):
    corpus = Corpus(fields_tuples)
    corpus.add_texts(texts)
    feature_fields = list(filter(lambda x: x[0] not in ['words', 'tags'],
                                 fields_tuples))
    if feature_fields:
        corpus.add_features(feature_fields,
                            options.prefix_min_length,
                            options.prefix_max_length,
                            options.suffix_min_length,
                            options.suffix_max_length)
    return PoSDataset(corpus)


class PoSDataset(Dataset):
    """Defines a dataset for PoS Tagging."""

    @staticmethod
    def sort_key(ex):
        return len(ex.words)

    def __init__(self, corpus, filter_pred=None):
        """Create a dataset from a list of Examples and Fields.

        Arguments:
            corpus: Corpus object.
            filter_pred (callable or None): Use only examples for which
                filter_pred(example) is True, or use all examples if None.
                Default is None.
        """
        # ensure that examples is not a generator
        examples = list(corpus)
        fields = corpus.attr_fields
        super().__init__(examples, fields, filter_pred)
