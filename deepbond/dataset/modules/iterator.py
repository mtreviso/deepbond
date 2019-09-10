# flake8: noqa: E501

import math
import random

from torchtext.data import Iterator, Dataset, Batch, BucketIterator


class BPTTIterator(Iterator):
    """Defines an iterator for language modeling tasks that use BPTT.

    Provides contiguous streams of examples together with targets that are
    one timestep further forward, for language modeling training with
    backpropagation through time (BPTT). Expects a Dataset with a single
    example and a single field called 'text' and produces Batches with text and
    target attributes.

    Attributes:
        dataset: The Dataset object to load Examples from.
        batch_size: Batch size.
        bptt_len: Length of sequences for backpropagation through time.
        sort_key: A key to use for sorting examples in order to batch together
            examples with similar lengths and minimize padding. The sort_key
            provided to the Iterator constructor overrides the sort_key
            attribute of the Dataset, or defers to it if None.
        train: Whether the iterator represents a train set.
        repeat: Whether to repeat the iterator for multiple epochs. Default: False.
        shuffle: Whether to shuffle examples between epochs.
        sort: Whether to sort examples according to self.sort_key.
            Note that shuffle and sort default to train and (not train).
        randomized_bptt_len: Whether to randomize the bptt_len between epochs. Will
            randomly increase/decrease from a normal distribution with std equal to 5.
            The final bptt_len is guaranteed to be always larger or equal to 5.
        device (str or torch.device): A string or instance of `torch.device`
            specifying which device the Variables are going to be created on.
            If left as default, the tensors will be created on cpu. Default: None.
    """

    def __init__(
            self, dataset, batch_size, bptt_len, randomized_bptt_len=False, **kwargs
    ):
        super(BPTTIterator, self).__init__(dataset, batch_size, **kwargs)
        self.bptt_len = bptt_len
        self.cur_bptt_len = bptt_len
        self.randomized_bptt_len = randomized_bptt_len
        self.field_name = self.get_unique_field_name(self.dataset.fields)
        self.batch_first = self.dataset.fields[self.field_name].batch_first

    def get_unique_field_name(self, fields):
        assert len(fields) == 1  # maybe remove this assert?
        return next(iter(fields.keys()))

    def set_random_bptt_len(self):
        self.cur_bptt_len = self.bptt_len
        if random.random() > 0.95:
            self.cur_bptt_len = self.bptt_len // 2
        self.cur_bptt_len = int(random.normalvariate(self.cur_bptt_len, 5))
        self.cur_bptt_len = max(10, self.cur_bptt_len)

    def __len__(self):
        return math.ceil((len(self.dataset[0].text) / self.batch_size - 1)
                         / self.cur_bptt_len)

    def prepare_text(self, text):
        """ text is a list of str """
        text_field = self.dataset.fields[self.field_name]
        text_field.eos_token = None  # this should be optional?

        nb_batches = math.ceil(len(text) / self.batch_size)
        nb_iters = nb_batches * self.batch_size
        text = text + [text_field.pad_token] * int(nb_iters - len(text))

        data = text_field.numericalize([text], device=self.device)
        data = data.view(self.batch_size, -1).t().contiguous()
        aux_fields = [(self.field_name, text_field), ('target', text_field)]
        aux_dataset = Dataset(self.dataset.examples, aux_fields)
        return data, aux_dataset

    def __iter__(self):
        text = getattr(self.dataset[0], self.field_name)
        data, dataset = self.prepare_text(text)
        while True:
            for i in range(0, len(self) * self.cur_bptt_len, self.cur_bptt_len):
                self.iterations += 1
                seq_len = min(self.cur_bptt_len, len(data) - i - 1)
                batch_text = data[i:i + seq_len]
                batch_target = data[i + 1:i + 1 + seq_len]
                if self.batch_first:
                    batch_text = batch_text.t().contiguous()
                    batch_target = batch_target.t().contiguous()
                yield Batch.fromvars(
                    dataset,
                    self.batch_size,
                    text=batch_text,
                    target=batch_target
                )
            if not self.repeat:
                return


class LazyIterator(Iterator):
    """
    Consume a generator for a specific number of steps (`buffer_size`), storing
    the examples in a buffer. The iterator will be built using this buffer.

    Args:
        buffer_size(int): The number of examples to be stored in the buffer.
            Default: batch_size * 1024
    """

    def __init__(self, *args, buffer_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = []
        if buffer_size is None:
            # minibatches will have the same size if buffer_size
            # is divisible by batch_size
            buffer_size = self.batch_size * 2 ** 10
        self.buffer_size = buffer_size
        self.batches = None

    def data(self):
        return iter(self.dataset)

    def clear_buffer(self):
        self.buffer.clear()

    def prepare_buffer(self):
        if self.sort:
            self.buffer.sort(key=self.sort_key)
        elif self.shuffle:
            buffer_size = range(len(self.buffer))
            self.buffer = [self.buffer[i] for i in self.random_shuffler(buffer_size)]

    def create_batches(self):
        self.batches = batch(self.buffer, self.batch_size, self.batch_size_fn)

    def consume_buffer(self):
        self.prepare_buffer()
        self.create_batches()
        for minibatch in self.batches:
            self.iterations += 1
            self._iterations_this_epoch += 1
            if self.sort_within_batch:
                if self.sort:
                    minibatch.reverse()
                else:
                    minibatch.sort(key=self.sort_key, reverse=True)
            yield Batch(minibatch, self.dataset, self.device)

    def __iter__(self):
        while True:
            self.init_epoch()
            self.clear_buffer()

            for ex in self.data():
                self.buffer.append(ex)
                if len(self.buffer) == self.buffer_size:
                    for batch in self.consume_buffer():
                        yield batch
                    self.clear_buffer()

            # in case the buffer is not empty
            if len(self.buffer) > 0:
                for batch in self.consume_buffer():
                    yield batch
                self.clear_buffer()

            if not self.repeat:
                return


class LazyBucketIterator(BucketIterator, LazyIterator):

    def create_batches(self):
        if self.sort:
            self.batches = batch(self.buffer,
                                 self.batch_size,
                                 self.batch_size_fn)
        else:
            self.batches = pool(self.buffer,
                                self.batch_size,
                                self.sort_key,
                                self.batch_size_fn,
                                random_shuffler=self.random_shuffler,
                                shuffle=self.shuffle,
                                sort_within_batch=self.sort_within_batch,
                                lookahead=self.buffer_size)


class LazyBPTTIterator(BPTTIterator, LazyIterator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_text_buffer = []

    def __len__(self):
        return self.get_len(self.buffer)

    def get_len(self, text_buffer):
        return math.ceil((len(text_buffer) / self.batch_size - 1) / self.cur_bptt_len)

    def clear_buffer(self):
        self.buffer.clear()
        self.prev_text_buffer.clear()

    def get_contiguous_buffer(self):
        text_buffer = []
        while (
            len(text_buffer) < self.buffer_size * self.cur_bptt_len
            and len(self.prev_text_buffer) > 0
        ):
            text_buffer.append(self.prev_text_buffer.pop(0))
        if len(text_buffer) == self.cur_bptt_len:
            return text_buffer
        for ex in self.buffer:
            text = getattr(ex, self.field_name)
            self.prev_text_buffer = []
            for w in text:
                if len(text_buffer) + 1 <= self.buffer_size * self.cur_bptt_len:
                    text_buffer.append(w)
                else:
                    self.prev_text_buffer.append(w)
        return text_buffer

    def prepare_text_buffer(self, text):
        """ text is a list of str """
        text_field = self.dataset.fields[self.field_name]
        text_field.eos_token = None  # this should be optional

        nb_batches = math.ceil(len(text) / self.batch_size)
        nb_iters = nb_batches * self.batch_size
        text = text + [text_field.pad_token] * int(nb_iters - len(text))

        data = text_field.numericalize([text], device=self.device)
        data = data.view(self.batch_size, -1).t().contiguous()
        aux_fields = [(self.field_name, text_field), ('target', text_field)]
        aux_dataset = Dataset(self.dataset.examples, aux_fields)
        return data, aux_dataset

    def consume_data(self, data, text_len):
        for i in range(0, text_len * self.cur_bptt_len, self.cur_bptt_len):
            self.iterations += 1
            seq_len = min(self.cur_bptt_len, len(data) - i - 1)
            batch_text = data[i:i + seq_len]
            batch_target = data[i + 1:i + 1 + seq_len]
            if self.batch_first:
                batch_text = batch_text.t().contiguous()
                batch_target = batch_target.t().contiguous()
            yield batch_text, batch_target

    def consume_buffer(self):
        cur_text_buffer = self.get_contiguous_buffer()
        data, dataset = self.prepare_text_buffer(cur_text_buffer)
        t_len = self.get_len(cur_text_buffer)
        for batch_text, batch_target in self.consume_data(data, t_len):
            kwargs = {self.field_name: batch_text, 'target': batch_target}
            yield Batch.fromvars(
                dataset,
                self.batch_size,
                **kwargs
            )

    def __iter__(self):
        while True:
            self.init_epoch()
            self.clear_buffer()

            if self.randomized_bptt_len:
                self.set_random_bptt_len()

            for ex in self.data():
                self.buffer.append(ex)
                if len(self.buffer) == self.buffer_size:
                    for batch in self.consume_buffer():
                        yield batch
                    self.clear_buffer()

            # in case the buffer is not empty
            if len(self.buffer) > 0:
                for batch in self.consume_buffer():
                    yield batch
                self.clear_buffer()

            if not self.repeat:
                return


def batch(data, batch_size, batch_size_fn=None):
    """Yield elements from data in chunks of batch_size."""
    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            return count
    minibatch = []
    size_so_far = 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
    if minibatch:
        yield minibatch


def pool(data, batch_size, key, batch_size_fn=lambda new, count, sofar: count,
         random_shuffler=None, shuffle=False, sort_within_batch=False,
         lookahead=100):
    """Sort within buckets, then batch, then shuffle batches.
    Partitions data into chunks of size lookahead*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.
    """
    if random_shuffler is None:
        random_shuffler = random.shuffle
    for p in batch(data, batch_size * lookahead, batch_size_fn):
        if sort_within_batch:
            p = sorted(p, key=key)
        p_batch = batch(p, batch_size, batch_size_fn)
        if shuffle:
            for b in random_shuffler(list(p_batch)):
                yield b
        else:
            for b in p_batch:
                yield b
