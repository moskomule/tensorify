""" Tensor decompositions
"""

from __future__ import annotations

import abc

import torch


class DecompositionBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def decompose(self, input, **kwargs) -> DecompositionBase:
        """ Decompose a given tensor

        :param input:
        :return:
        """

    @abc.abstractmethod
    def compose(self, **kwargs) -> torch.Tensor:
        """ Compose a tensor from `DecompositionBase`

        :param inputs:
        :return:
        """

    def __add__(self, other):
        # element-wise summation
        raise NotImplementedError

    def __mul__(self, other):
        # element-wise multiplication
        raise NotImplementedError

    def norm(self):
        # norm
        raise NotImplementedError

    @property
    def size(self):
        raise NotImplementedError

    @property
    def rank(self):
        raise NotImplementedError
