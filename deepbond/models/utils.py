import copy

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


def unmask(tensor, mask, cut_length=0):
    """
    Unmask a tensor and convert it back to a list of lists.

    Args:
        tensor (torch.Tensor): tensor with shape (bs, max_len, ...)
        mask (torch.Tensor): tensor with shape (bs, max_len) where 1 (or True)
            indicates a valid position, and 0 (or False) otherwise
        cut_length (int): remove the last `cut_length` elements from the tensor.
            In practice, the lengths calculated from the mask are going to be
            subtracted by `cut_length`. This is useful when you have <bos> and
            <eos> tokens in your words field and the mask was computed with
            words != <pad>. Default is 0, i.e., no cut
    Returns:
         a list of lists with variable length
    """
    lengths = mask.int().sum(dim=-1)
    # if the mask was calculated using words, then we subtract cut_length
    # in practice: to remove the size of the <bos> and <eos> tokens
    # which are already removed from the tensor in the forward pass but not
    # from the mask
    if cut_length > 0:
        lengths -= cut_length
    lengths = lengths.tolist()
    return [x[:lengths[i]].tolist() for i, x in enumerate(tensor)]


def unroll(list_of_lists, rec=False):
    """
    Unroll a list of lists.

    Args:
        list_of_lists (list): a list that contains lists
        rec (bool): unroll recursively
    Returns:
        a single list
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


def neighbours_mask(size, window_size):
    """Mask for neighbour positions.

    Args:
        size(int): squared tensor size
        window_size(int): how many elements to be considered as valid around
            the ith element (including ith).
    """
    z = torch.ones(size, size, dtype=torch.uint8)
    mask = (torch.triu(z, diagonal=1 + window_size // 2) +
            torch.tril(z, diagonal=- window_size // 2))
    return z - mask


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


def apply_packed_sequence(rnn, padded_sequences, lengths, hidden=None):
    """
    Code adapted from Unbabel OpenKiwi.
    Runs a forward pass of embeddings through an rnn using packed sequence.

    Args:
       rnn: The RNN that that we want to compute a forward pass with.
       padded_sequences (FloatTensor b x seq x dim): A batch of sequence seqs.
       lengths (LongTensor batch): The length of each sequence in the batch.
       hidden (FloatTensor, optional): hidden state for the rnn.
    Returns:
       output: The output of the RNN `rnn` with input `embedding`
    """
    # Sort Batch by sequence length
    total_length = padded_sequences.size(1)  # Get the max sequence length
    lengths_sorted, permutation = torch.sort(lengths, descending=True)
    padded_sequences_sorted = padded_sequences[permutation]

    # Use Packed Sequence
    embedding_packed = pack(
        padded_sequences_sorted, lengths_sorted, batch_first=True
    )
    outputs_packed, hidden = rnn(embedding_packed, hidden)
    outputs_sorted, _ = unpack(
        outputs_packed, batch_first=True, total_length=total_length
    )

    # Restore original order
    _, permutation_rev = torch.sort(permutation, descending=False)
    outputs = outputs_sorted[permutation_rev]
    hidden[0] = hidden[0][permutation_rev]
    hidden[1] = hidden[1][permutation_rev]
    return outputs, hidden
