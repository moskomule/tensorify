import torch

from .operations import __all__ as _operations
from .tensorify import register_as_methods

__all__ = ["torch"]
__all__ += _operations

register_as_methods()
