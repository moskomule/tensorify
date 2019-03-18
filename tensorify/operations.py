from typing import Sequence, Tuple, Optional

import torch

__all__ = ["is_vector", "is_matrix",
           "product", "multilinier_product", "contracted_product", "outer_product", "kronecker_product"]


def is_vector(input: torch.Tensor):
    return input.dim() == 1


def is_matrix(input: torch.Tensor):
    return input.dim() == 2


def matricization(input: torch.Tensor, mode: int):
    r""" Mode-n matricization of the input tensor as :math:`\mathbb{R}^{I_n\times I_1\dotsI_{n-1}I_{n+1}\dots I_N}`

    :param input:
    :param mode:
    :return:
    """
    if input.dim() <= mode:
        raise ValueError(f"`mode` is expected to be < {input.dim()} but got {mode}")
    if input.dim() > 26:
        raise RuntimeError("Dimension larger than 26 is not supported")
    input_alp = "abcdefghijklmnopqrstuvwxyz"[:input.dim()]
    mode_alp = input_alp[mode]
    new_input_alp = mode_alp + input_alp.replace(mode_alp, "")
    return torch.einsum(f"{input_alp}->{new_input_alp}",
                        # to make sure the contiguity
                        input).reshape(input.size(mode), -1)


def canonical_matricization(input: torch.Tensor, mode: int):
    r""" Mode-n canonical matricization of the input tensor as :math:`\mathbb{R}^{I_1\dotsI_n\timesI_{n+1}\dots I_N}`

    :param input:
    :param mode:
    :return:
    """
    if input.dim() - 1 <= mode:
        raise ValueError(f"`mode` is expected to be < {input.dim() - 1} but got {mode}")
    input_size = torch.tensor(input.size())
    return input.view(torch.prod(input_size[:mode + 1]), torch.prod(input_size[mode + 1:]))


def product(input: torch.Tensor,
            other: torch.Tensor,
            mode: int) -> torch.Tensor:
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
                        factors: Sequence[torch.Tensor]) -> torch.Tensor:
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
                       other: torch.Tensor,
                       dims: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """ Mode-(N, 1) contracted product of two tensors

    :param input:
    :param other:
    :param dims:
    :return:
    """
    if dims is None:
        dims = [input.dim() - 1, 0]
    if len(dims) != 2:
        raise ValueError(f"`dims` should be a tuple of a pair of integers")
    if input.size(dims[0]) != other.size(dims[1]):
        raise ValueError("-1st mode of `input` and 0th mode of `other` should be same")
    return torch.tensordot(input, other, [[dims[0]], [dims[1]]])


def outer_product(input: torch.Tensor,
                  other: torch.Tensor) -> torch.Tensor:
    """ Outer product of two tensors

    :param input:
    :param other:
    :return:
    """
    if input.dim() > 13 or other.dim() > 13:
        raise RuntimeError("Dimension larger than 13 is not supported")
    input_alp = "abcdefghijklm"[:input.dim()]
    other_alp = "nopqrstuvwxyz"[:other.dim()]
    return torch.einsum(f"{input_alp},{other_alp}->{input_alp}{other_alp}",
                        input,
                        other)


def kronecker_product(input: torch.Tensor,
                      other: torch.Tensor) -> torch.Tensor:
    """ Left Kronecker product of two tensors

    :param input:
    :param other:
    :return:
    """
    if input.dim() != other.dim():
        raise ValueError("The dimensions of `input` and `other` should be same")
    input_size = torch.tensor(input.size())
    other_size = torch.tensor(other.size())

    # input_size (a, b, c) -> (a, 1, b, 1, c, 1)
    new_input_size = torch.stack([input_size, torch.ones_like(input_size)], dim=1).view(-1).tolist()
    new_other_size = torch.stack([torch.ones_like(other_size), other_size], dim=1).view(-1).tolist()
    expected_size = (input_size * other_size).tolist()
    return (input.view(new_input_size) * other.view(new_other_size)).view(expected_size)
