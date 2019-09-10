from torchtext.data import Dataset


class LazyDataset(Dataset):

    def __init__(self, examples, fields, filter_pred=lambda x: True):
        self.examples = examples
        self.fields = dict(fields)
        self.filter_pred = filter_pred
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

    def __iter__(self):
        for ex in self.examples:
            if self.filter_pred(ex):
                yield ex
