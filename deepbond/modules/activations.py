import math

import torch


def gelu(x):
    """gelu activation function copied from pytorch-pretrained-BERT."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    """swusg activation function copied from pytorch-pretrained-BERT."""
    return x * torch.sigmoid(x)
