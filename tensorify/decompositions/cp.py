from typing import Iterable, Optional

import torch

from tensorify.operations import is_vector, matricization, khatri_rao_product
from .meta import DecompositionBase


class CP(DecompositionBase):
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
                  method: str = "als",
                  **kwargs):
        """ Decompose a given tensor using a given method.

        :param input:
        :param method:
        :param kwargs:
        :return:
        """

        if method == "als":
            core, factors = CP.als(input, **kwargs)
            self.core = core
            self.factors = factors
            return core, factors
        else:
            raise NotImplementedError

    def compose(self):
        """ Compose a tensor from `self.core` and `self.factors`

        :return:
        """

        if self.core is None or self.factors is None:
            raise RuntimeError("Tensor is not decomposed yet!")
        return CP.cp_compose(self.core, self.factors)

    @property
    def size(self):
        return None if self.core is None else self.core.size()

    @property
    def rank(self):
        return None if self.factors is None else len(self.factors)

    @staticmethod
    def cp_compose(core: torch.Tensor,
                   factors: Iterable[torch.Tensor]) -> torch.Tensor:
        """ Compose a tensor from given core and factor tensors

        :param core:
        :param factors:
        :return:
        """

        # rank: z
        # factors: az, bz, cz...
        alph = "abcdefghijklmnopqrstuvwxy"[:len(factors)]
        alph = "z," + "".join([i + "z," for i in alph])[:-1]
        # z,az,bz,... -> abc...
        return torch.einsum(alph,
                            *[core, *factors])

    @staticmethod
    def als(input: torch.Tensor,
            rank: int,
            eps: float = 1e-4,
            max_iter: int = 100):
        """ Naive implementation of ALS for CP decomposition

        :param input:
        :param rank:
        :param eps:
        :param max_iter:
        :return:
        """

        if rank > min(input.size()):
            raise ValueError(f"`rank` is expected to be < min(input.size()) {min(input.size())}")
        # input: IxJxK
        # factors: IxR, JxR, KxR
        factors = [input.new_empty(i, rank).normal_() for i in input.size()]
        dim = len(factors)
        for _ in range(max_iter):
            for i in range(dim):
                # suppose i = 0
                # _self: IxR, _next: JxR, _nextnext: KxR
                _self = factors[i]
                _next = factors[(i + 1) % dim]
                _nextnext = factors[(i + 2) % dim]
                # JxR,KxR -> JKxR
                krp = khatri_rao_product(_nextnext, _next, 1)
                # todo: need to be pinverse? inv: RxR
                inv = (_nextnext.t().matmul(_nextnext) * _next.t().matmul(_next)).pinverse()
                new_factor = matricization(input, i).matmul(krp).matmul(inv)
                norm = new_factor.norm(dim=0)
                factors[i] = new_factor / norm
            if (CP.cp_compose(norm, factors) - input).norm() < eps:
                break
        return norm, factors
