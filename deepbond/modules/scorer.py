import math

import torch
from torch import nn

from deepbond.models.utils import make_mergeable_tensors


class Scorer(nn.Module):
    """Score function for Attention module.

    Args:
        scaled (bool): wheter to scale scores by `sqrt(hidden_size)` as
            proposed by "Attention is All You Need" paper.
    """

    def __init__(self, scaled=True):
        super().__init__()
        self.scaled = scaled

    def scale(self, hidden_size):
        """Denominator for scaling the scores.

        Args:
            hidden_size(int): size of input vector

        Returns:
            int: sqrt(hidden_size) if `scaled` is True, 1 otherwise
        """
        if self.scaled:
            return math.sqrt(hidden_size)
        return 1

    def forward(self, query, keys):
        """Computes scores for each key of size n given the queries of size m.

        The three dots (...) represent any other dimensions, such as the
        number of heads (useful if you use a multi head attention).

        Args:
            query (torch.FloatTensor): query matrix (bs, ..., target_len, m)
            keys (torch.FloatTensor): keys matrix (bs, ..., source_len, n)

        Returns:
            torch.FloatTensor: matrix representing scores between souce words
                and target words: (bs, ..., target_len, source_len)
        """
        raise NotImplementedError


class DotProductScorer(Scorer):
    """Implements DotProduct function for attention.
       Query and keys should have the same size.
    """

    def forward(self, query, keys):
        # in DotProduct the keys and query vector should have the same size
        assert keys.shape[-1] == query.shape[-1]
        scale = self.scale(keys.shape[-1])

        # using matmul instead of einsum:
        # score = torch.matmul(query, keys.transpose(-1, -2))

        # b = batch size
        # t = target length
        # s = source length
        # h = hidden size
        score = torch.einsum('b...th,b...sh->b...ts', [query, keys])
        return score / scale


class GeneralScorer(Scorer):
    """Implement GeneralScorer (aka Multiplicative) for attention"""

    def __init__(self, query_size, key_size, **kwargs):
        super().__init__(**kwargs)
        self.W = nn.Parameter(torch.Tensor(key_size, query_size))
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, query, keys):
        scale = self.scale(keys.shape[-1])
        # score = torch.matmul(torch.matmul(query, self.W.t()), keys.transpose(-1, -2))  # NOQA
        score = torch.einsum('b...tm,nm,b...sn->b...ts', [query, self.W, keys])
        return score / scale


class SelfAdditiveScorer(Scorer):
    """
    This is a special case for AdditiveScorer when query=key (self attention).
    Its implementation is based on related works in order to this work to be
    comparable with them. Otherwise, you can use OperationScorer(op='add')
    """
    def __init__(self, vector_size, attn_hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.W = nn.Parameter(torch.Tensor(attn_hidden_size, vector_size))
        self.b = nn.Parameter(torch.Tensor(attn_hidden_size))
        self.v = nn.Parameter(torch.Tensor(1, attn_hidden_size))
        self.activation = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.W.shape[1])
        nn.init.uniform_(self.b, -bound, bound)
        nn.init.kaiming_uniform_(self.v, a=math.sqrt(5))

    def forward(self, query, keys):
        # keys == query
        scale = self.scale(keys.shape[-1])
        # x = torch.matmul(query, self.W.t()) + self.b
        x = torch.einsum('b...tm,hm->b...th', [query, self.W]) + self.b
        x = self.activation(x)
        # score = torch.matmul(x, self.v.t()).squeeze(-1)
        score = torch.einsum('b...th,oh->b...to', [x, self.v]).squeeze(-1)
        score = score.unsqueeze(1)
        return score / scale


class OperationScorer(Scorer):
    """Base class for ConcatScorer and AdditiveScorer"""

    def __init__(self, query_size, key_size, attn_hidden_size, op='concat',
                 activation=nn.Tanh, **kwargs):
        super().__init__(**kwargs)
        assert op in ['concat', 'add', 'mul']
        self.op = op
        self.activation = activation()
        self.W1 = nn.Parameter(torch.Tensor(attn_hidden_size, key_size))
        self.W2 = nn.Parameter(torch.Tensor(attn_hidden_size, query_size))
        if self.op == 'concat':
            self.v = nn.Parameter(torch.Tensor(1, 2 * attn_hidden_size))
        else:
            self.v = nn.Parameter(torch.Tensor(1, attn_hidden_size))
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v, a=math.sqrt(5))

    def f(self, x1, x2):
        """Perform an operation on x1 and x2"""
        if self.op == 'add':
            x = x1 + x2
        elif self.op == 'mul':
            x = x1 * x2
        else:
            x = torch.cat((x1, x2), dim=-1)
        return self.activation(x)

    def forward(self, query, keys):
        scale = self.scale(keys.shape[-1])
        # x1 = torch.matmul(keys, self.W1.t())
        # x2 = torch.matmul(query, self.W2.t())
        x1 = torch.einsum('b...tm,hm->b...th', [query, self.W2])
        x2 = torch.einsum('b...sn,hn->b...sh', [keys, self.W1])
        x1, x2 = make_mergeable_tensors(x1, x2)
        # score = torch.matmul(self.f(x1, x2), self.v.t())
        score = torch.einsum('b...tsh,oh->b...tso', [self.f(x1, x2), self.v])
        score = score.squeeze(-1)
        return score / scale


class MLPScorer(Scorer):
    """MultiLayerPerceptron Scorer with variable nb of layers and neurons."""

    def __init__(
        self,
        query_size,
        key_size,
        layer_sizes=None,
        activation=nn.Tanh,
        **kwargs
    ):
        super().__init__(**kwargs)
        if layer_sizes is None:
            layer_sizes = [(query_size + key_size) // 2]
        input_size = query_size + key_size  # concatenate query and keys
        output_size = 1  # produce a scalar for each alignment
        layer_sizes = [input_size] + layer_sizes + [output_size]
        sizes = zip(layer_sizes[:-1], layer_sizes[1:])
        layers = []
        for n_in, n_out in sizes:
            layers.append(
                nn.Sequential(
                    nn.Linear(n_in, n_out),
                    activation()
                )
            )
            # layers.append(nn.Linear(n_in, n_out))
            # layers.append(activation())
        self.layers = nn.ModuleList(layers)

    def forward(self, query, keys):
        x_query, x_keys = make_mergeable_tensors(query, keys)
        x = torch.cat((x_query, x_keys), dim=-1)
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(-1)  # remove last dimension


if __name__ == '__main__':
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    batch_size = 4
    source_len = 7
    target_len = 3
    query_size = 10
    keys_size = 20
    attn_size = 15

    # query vectors
    q = torch.randn(batch_size, target_len, query_size)

    # keys vectors (a key vector for each encoder word)
    ks = torch.randn(batch_size, source_len, keys_size)

    # keys vectors with same size as query vectors
    kq = torch.randn(batch_size, source_len, query_size)

    out = DotProductScorer()(q, kq)
    assert list(out.shape) == [batch_size, q.shape[1], kq.shape[1]]

    out = GeneralScorer(query_size, keys_size)(q, ks)
    assert list(out.shape) == [batch_size, q.shape[1], ks.shape[1]]

    out = OperationScorer(query_size, keys_size, attn_size, op='add')(q, ks)
    assert list(out.shape) == [batch_size, q.shape[1], ks.shape[1]]

    out = OperationScorer(query_size, keys_size, attn_size, op='mul')(q, ks)
    assert list(out.shape) == [batch_size, q.shape[1], ks.shape[1]]

    out = OperationScorer(query_size, keys_size, attn_size, op='concat')(q, ks)
    assert list(out.shape) == [batch_size, q.shape[1], ks.shape[1]]

    out = OperationScorer(query_size, query_size, attn_size, op='add')(q, q)
    assert list(out.shape) == [batch_size, q.shape[1], q.shape[1]]

    out = MLPScorer(
        query_size, keys_size, layer_sizes=[10, 5, 5], activation=nn.Sigmoid
    )(q, ks)
    assert list(out.shape) == [batch_size, q.shape[1], ks.shape[1]]

    out = MLPScorer(query_size, keys_size, layer_sizes=[10, 5, 5])(q, ks)
    assert list(out.shape) == [batch_size, q.shape[1], ks.shape[1]]
