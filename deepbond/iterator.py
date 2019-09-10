from torchtext.data import BucketIterator
import torch


def build(dataset, device, batch_size, is_train):
    device = None if device is None else torch.device(device)
    iterator = BucketIterator(
        dataset=dataset,
        batch_size=batch_size,
        repeat=False,
        # sorts the data within each minibatch in decreasing order according
        # set to true if you want use pack_padded_sequences
        sort_key=dataset.sort_key,
        sort=False,
        sort_within_batch=True,
        # shuffle batches
        shuffle=is_train,
        device=device,
        train=is_train
    )
    return iterator
