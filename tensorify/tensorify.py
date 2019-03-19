""" Contains some utilities
"""

import torch

from tensorify import operations


def register_as_methods():
    # register operations as Tensor's methods
    for name in operations.__all__:
        setattr(torch.Tensor, name, getattr(operations, name))
