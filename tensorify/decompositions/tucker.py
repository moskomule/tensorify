from __future__ import annotations

from typing import Optional, Iterable

import torch

from tensorify.operations import is_vector, multilinier_product, direct_sum, kronecker_product, khatri_rao_product
from .meta import DecompositionBase


class Tucker(DecompositionBase):
    def __init__(self,
                 core: Optional[torch.Tensor] = None,
                 factors: Optional[Iterable[torch.Tensor]] = None):
        """ CP format a.k.a. Canonical Polyadic decomposition

        :param core:
        :param factors:
        """

        if core is not None and not is_vector(core):
            raise RuntimeError("`core` is expected to be a vector")
        if not (core is None and factors is None) and (len(core) != len(factors)):
            raise RuntimeError("")
        self.core = core
        self.factors = factors

    def decompose(self,
                  input: torch.Tensor,
                  method: str = "hosvd",
                  **kwargs):
        """ Decompose a given tensor using a given method.

        :param input:
        :param method:
        :param kwargs:
        :return:
        """

        if method == "hosvd":
            return Tucker.hosvd(input, **kwargs)
        else:
            raise NotImplementedError

    def compose(self):
        """ Compose a tensor from `self.core` and `self.factors`

        :return:
        """

        if self.core is None or self.factors is None:
            raise RuntimeError("Tensor is not decomposed yet!")
        Tucker.tucker_composition(self.core, self.factors)

    @staticmethod
    def tucker_composition(core: torch.Tensor,
                           factors: Iterable[torch.Tensor]):
        return multilinier_product(core, factors)

    @staticmethod
    def hosvd(input: torch.Tensor, **kwargs):
        raise NotImplementedError

    @staticmethod
    def sequentially_truncated_hosvd(input: torch.Tensor,
                                     eps: float = 1e-2):
        raise NotImplementedError

    # Some operations

    def __add__(self, other: Tucker):
        if not isinstance(other, Tucker):
            raise RuntimeError(f"`other` is expected to be Tucker, but got {type(other)}")
        if self.core is None or self.factors is None:
            raise RuntimeError("Tensor is not decomposed yet!")
        if other.core is None or other.factors is None:
            raise RuntimeError("`other` is not decomposed yet!")
        if len(self.factors) != len(other.core):
            raise RuntimeError("Lengths of `self.factor` and `other.factor` are expected to be same")

        core = direct_sum(self.core, other.core)
        factors = [torch.cat([x, y], dim=1) for (x, y) in zip(self.factors, other.factors)]
        return Tucker(core, factors)

    def __neg__(self):
        return Tucker(-self.core, self.factors)

    def __mul__(self, other: torch.Tensor):
        return self.hadamard_product(other)

    def kronecker_product(self, other: Tucker):
        return Tucker(kronecker_product(self.core, other.core),
                      [kronecker_product(x, y) for (x, y) in zip(self.factors, other.factors)])

    def hadamard_product(self, other: Tucker):
        return Tucker(kronecker_product(self.core, other.core),
                      [khatri_rao_product(x, y, 0) for (x, y) in zip(self.factors, other.factors)])

    def inner_product(self, other: Tucker):
        left = multilinier_product(self.core,
                                   [x.t().matmul(y) for (x, y) in zip(self.factors, other.factors)])
        return (left * other.core).sum()

    def norm(self):
        return self.core.norm()
