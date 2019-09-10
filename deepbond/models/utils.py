import copy

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def indexes_to_words(indexes, itos):
    """
    Transofrm indexes to words using itos list
    :param indexes: list of lists of ints
    :param itos: list mapping integer to string
    :return: list of lists of strs
    """
    words = []
    for sample in indexes:
        words.append([itos[i] for i in sample])
    return words


def unmask(tensor, mask):
    """
    Unmask a tensor and convert it back to a list of lists.
    :param tensor: a torch.tensor object
    :param mask: a torch.tensor object with 1 indicating a valid position
                 and 0 elsewhere
    :return: a list of lists with variable length
    """
    lengths = mask.int().sum(dim=-1).tolist()
    return [x[:lengths[i]].tolist() for i, x in enumerate(tensor)]


def unroll(list_of_lists, rec=False):
    """
    :param list_of_lists: a list that contains lists
    :param rec: unroll recursively
    :return: a flattened list
    """
    if not isinstance(list_of_lists[0], (np.ndarray, list)):
        return list_of_lists
    new_list = [item for l in list_of_lists for item in l]
    if rec and isinstance(new_list[0], (np.ndarray, list)):
        return unroll(new_list, rec=rec)
    return new_list


def clones(module, N):
    """Produce N identical layers."""
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """Mask out subsequent positions.

    Args:
        size(int): squared tensor size
    """
    return torch.tril(torch.ones(size, size, dtype=torch.uint8))


def sequence_mask(lengths, max_len=None):
    """Creates a boolean mask from sequence lengths.

    Args:
        lengths (torch.LongTensor): lengths with shape (bs,)
        max_len (int, optional): max sequence length.
            if None it will be setted to lengths.max()
    """
    if max_len is None:
        max_len = lengths.max()
    aranges = torch.arange(max_len).repeat(lengths.shape[0], 1)
    aranges = aranges.to(lengths.device)
    return aranges < lengths.unsqueeze(1)


def unsqueeze_as(tensor, as_tensor, dim=-1):
    """Expand new dimensions based on a template tensor along `dim` axis."""
    x = tensor
    while x.dim() < as_tensor.dim():
        x = x.unsqueeze(dim)
    return x


def make_mergeable_tensors(t1, t2):
    """Expand a new dimension in t1 and t2 and expand them so that both
    tensors will have the same number of timesteps.

    Args:
        t1 (torch.Tensor): tensor with shape (bs, ..., m, d1)
        t2 (torch.Tensor): tensor with shape (bs, ..., n, d2)

    Returns:
        torch.Tensor: (bs, ..., m, n, d1)
        torch.Tensor: (bs, ..., m, n, d2)
    """
    assert t1.dim() == t2.dim()
    assert t1.dim() >= 3
    assert t1.shape[:-2] == t2.shape[:-2]
    # new_shape = [-1, ..., m, n, -1]
    new_shape = [-1 for _ in range(t1.dim() + 1)]
    new_shape[-3] = t1.shape[-2]  # m
    new_shape[-2] = t2.shape[-2]  # n
    # (bs, ..., m, d1) -> (bs, ..., m, 1, d1) -> (bs, ..., m, n, d1)
    new_t1 = t1.unsqueeze(-2).expand(new_shape)
    # (bs, ..., n, d2) -> (bs, ..., 1, n, d2) -> (bs, ..., m, n, d2)
    new_t2 = t2.unsqueeze(-3).expand(new_shape)
    return new_t1, new_t2


def apply_packed_sequence(rnn, embedding, lengths):
    """
    Code from Unbabel OpenKiwi

    Runs a forward pass of embeddings through an rnn using packed sequence.
    Args:
       rnn: The RNN that that we want to compute a forward pass with.
       embedding (FloatTensor b x seq x dim): A batch of sequence embeddings.
       lengths (LongTensor batch): The length of each sequence in the batch.

    Returns:
       output: The output of the RNN `rnn` with input `embedding`
    """
    # Sort Batch by sequence length
    lengths_sorted, permutation = torch.sort(lengths, descending=True)
    embedding_sorted = embedding[permutation]

    # Use Packed Sequence
    embedding_packed = pack(embedding_sorted, lengths_sorted, batch_first=True)
    outputs_packed, _ = rnn(embedding_packed)
    outputs_sorted, _ = unpack(outputs_packed, batch_first=True)

    # Restore original order
    _, permutation_rev = torch.sort(permutation, descending=False)
    outputs = outputs_sorted[permutation_rev]
    return outputs
