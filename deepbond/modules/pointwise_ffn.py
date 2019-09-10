from torch import nn

from deepbond.modules.activations import gelu, swish

ACT2FN = {
    'gelu': gelu,
    'swish': swish,
    'relu': nn.functional.relu,
    'selu': nn.functional.selu,
}


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation.

    Args:
        hidden_size (int): input dimension
        inner_hidden_size (int): hidden ffn dimension
        activation_fn (str): nonlinearity function `gelu`, `relu` or `selu`
            (default: gelu)
    """

    def __init__(self, hidden_size, inner_hidden_size, activation_fn='gelu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.activation_fn = ACT2FN[activation_fn]
        self.w_1 = nn.Linear(hidden_size, inner_hidden_size)
        self.w_2 = nn.Linear(inner_hidden_size, hidden_size)

    def forward(self, x):
        return self.w_2(self.activation_fn(self.w_1(x)))
