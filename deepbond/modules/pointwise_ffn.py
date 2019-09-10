from torch import nn


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation.

    Args:
        hidden_size (int): input dimension
        inner_hidden_size (int): hidden ffn dimension
        activation_fn (function): nonlinearity function (default: relu)
    """

    def __init__(
        self,
        hidden_size,
        inner_hidden_size,
        dropout=0.1,
        activation_fn=nn.functional.relu,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.w_1 = nn.Linear(hidden_size, inner_hidden_size)
        self.w_2 = nn.Linear(inner_hidden_size, hidden_size)

    def forward(self, x):
        return self.w_2(self.activation_fn(self.w_1(x)))
