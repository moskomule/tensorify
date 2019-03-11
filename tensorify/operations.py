from typing import Sequence

import torch

__all__ = ["is_vector", "is_matrix",
           "product", "multilinier_product", "contracted_product", "outer_product", "kronecker_product"]


def is_vector(input: torch.Tensor):
    return input.dim() == 1


def is_matrix(input: torch.Tensor):
    return input.dim() == 2


def product(input: torch.Tensor,
            other: torch.Tensor,
            mode: int):
    r""" Mode-n product of a tensor :math:`\mathrm{input}\in\mathbb{R}^{I_1\times I_2\times\dots\timesI_N}` and
    a matrix :math:`\mathrm{other}\in\mathbb{J\times I_n}` or a vector :math:`\mathrm{other}\in\mathbb{I_n}`.

    :param input:
    :param other:
    :param mode:
    :return:
    """

    if input.dim() <= mode:
        raise ValueError(f"`mode` is expected to be <= {input.dim()} but got {mode}")
    if not (is_vector(other) or is_matrix(other)):
        raise ValueError(f"`other` is expected to be vector or matrix")
    if input.size(mode) != other.size(-1):
        raise ValueError(f"Size mismatch")
    if is_vector(other):
        other = other.view(1, -1)
    return input.transpose(mode, -1).matmul(other.t()).transpose_(mode, -1).squeeze_(mode)


def multilinier_product(core: torch.Tensor,
                        factors: Sequence[torch.Tensor]):
    r""" Multilinier product of a core tensor :math:`\mathrm{core}` and factor matrices
    :math:`\mathrm{factor}_0, \mathrm{factor}_1, \dots,`,

    :param core:
    :param factors:
    :return:
    """
    if core.dim() != len(factors):
        raise ValueError("Dimension of `core` and length of `factors` should be equal")
    for i, mat in enumerate(factors):
        if not is_matrix(mat):
            raise ValueError(f"{i}th element of `factors` is not matrix")
        core = product(core, mat, i)
    return core


def contracted_product(input: torch.Tensor,
                       other: torch.Tensor):
    """ Mode-(N, 1) contracted product of two tensors

    :param input:
    :param other:
    :return:
    """
    if input.size(-1) != other.size(0):
        raise ValueError(f"-1st mode of `input` and 0th mode of `other` should be same")
    out_shape = input.size()[:-1] + other.size()[1:]
    return input.view(-1, input.size(-1)).matmul(other.view(other.size(0), -1)).view(out_shape)


def outer_product(input: torch.Tensor,
                  other: torch.Tensor):
    """ Outer product of two tensors

    :param input:
    :param other:
    :return:
    """
    if input.dim() > 13 or other.dim() > 13:
        raise RuntimeError("Dimension larger than 26 is not supported")
    input_alp = "abcdefghijklm"[:input.dim()]
    other_alp = "nopqrstuvwxyz"[:other.dim()]
    return torch.einsum(f"{input_alp},{other_alp}->{input_alp}{other_alp}",
                        input,
                        other)


def kronecker_product(input: torch.Tensor,
                      target: torch.Tensor):
    """ Left Kronecker product of two tensors

    :param input:
    :param target:
    :return:
    """
    raise NotImplementedError
