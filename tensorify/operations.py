from typing import Sequence, Tuple, Optional

import torch

__all__ = ["is_vector", "is_matrix",
           "product", "multilinier_product", "contracted_product", "outer_product", "kronecker_product",
           "khatri_rao_product", "direct_sum"]


def is_vector(input: torch.Tensor):
    return input.dim() == 1


def is_matrix(input: torch.Tensor):
    return input.dim() == 2


def matricization(input: torch.Tensor,
                  mode: int):
    r""" Mode-n matricization of the input tensor as :math:`\mathbb{R}^{I_n\times I_1\dotsI_{n-1}I_{n+1}\dots I_N}`

    :param input:
    :param mode:
    :return:
    """
    if input.dim() <= mode:
        raise ValueError(f"`mode` is expected to be < {input.dim()}")
    if input.dim() > 26:
        raise RuntimeError("Dimension larger than 26 is not supported")
    input_alp = "abcdefghijklmnopqrstuvwxyz"[:input.dim()]
    mode_alp = input_alp[mode]
    new_input_alp = mode_alp + input_alp.replace(mode_alp, "")
    return torch.einsum(f"{input_alp}->{new_input_alp}",
                        # to make sure the contiguity
                        input).reshape(input.size(mode), -1)


def canonical_matricization(input: torch.Tensor,
                            mode: int):
    r""" Mode-n canonical matricization of the input tensor as :math:`\mathbb{R}^{I_1\dotsI_n\timesI_{n+1}\dots I_N}`

    :param input:
    :param mode:
    :return:
    """
    if input.dim() - 1 <= mode:
        raise ValueError(f"`mode` is expected to be < {input.dim() - 1} ")
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
        raise ValueError(f"Sizes at {mode} th mode of `input` and at -1 the mode of `other` are expected to be same")
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
        raise ValueError("Dimension of `core` and length of `factors` are expected to be same")
    for i, mat in enumerate(factors):
        if not is_matrix(mat):
            raise ValueError(f"{i}th element of `factors` is expected to be a matrix")
        core = product(core, mat, i)
    return core


def contracted_product(input: torch.Tensor,
                       other: torch.Tensor,
                       modes: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """ Mode-(N, 1) contracted product of two tensors

    :param input:
    :param other:
    :param modes:
    :return:
    """
    if modes is None:
        modes = [input.dim() - 1, 0]
    if len(modes) != 2:
        raise ValueError(f"`modes` is expected to be a tuple of a pair of integers")
    if input.size(modes[0]) != other.size(modes[1]):
        raise ValueError("-1st mode of `input` and 0th mode of `other` are expected to be same")
    return torch.tensordot(input, other, [[modes[0]], [modes[1]]])


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


def _kkr_product(input: torch.Tensor,
                 other: torch.Tensor,
                 mode: Optional[int] = None) -> torch.Tensor:
    """ backend of `kronecker_product` nad `khatri_rao_product`.

    :param input:
    :param other:
    :param mode:
    :return:
    """
    if input.dim() != other.dim():
        raise ValueError("Dimensions of `input` and `other` are expected to be same")
    if (mode is not None) and (input.size(mode) != other.size(mode)):
        raise ValueError(f"Sizes at {mode}th mode are expected to be same")

    input_size = torch.tensor(input.size())
    other_size = torch.tensor(other.size())

    # input_size (a, b, c) -> (a, 1, b, 1, c, 1)
    new_input_size = torch.stack([input_size, torch.ones_like(input_size)], dim=1).view(-1).tolist()
    new_other_size = torch.stack([torch.ones_like(other_size), other_size], dim=1).view(-1).tolist()
    output_size = (input_size * other_size)
    if mode is not None:
        new_input_size.pop(2 * mode + 1)
        new_other_size.pop(2 * mode - 1)
        output_size[mode] / input.size(mode)

    return (input.view(new_input_size) * other.view(new_other_size)).view(output_size.tolist())


def khatri_rao_product(input: torch.Tensor,
                       other: torch.Tensor,
                       mode: int) -> torch.Tensor:
    """ Left Khatri-Rao product of tensors

    :param input:
    :param other:
    :return:
    """
    if input.dim() <= mode:
        raise ValueError(f"`mode` should be smaller than {input.mode()}")
    return _kkr_product(input, other, mode)


def kronecker_product(input: torch.Tensor,
                      other: torch.Tensor) -> torch.Tensor:
    """ Left Kronecker product of two tensors

    :param input:
    :param other:
    :return:
    """
    return _kkr_product(input, other, None)


def direct_sum(input: torch.Tensor,
               other: torch.Tensor,
               mode: Optional[int] = None) -> torch.Tensor:
    """ Direct sum of two tensors

    :param input:
    :param other:
    :param mode:
    :return:
    """
    if input.dim() != other.dim():
        raise ValueError("Dimensions of `input` and `other` are expected to be same")
    if mode is not None and (input.size(mode) != other.size(mode)):
        raise ValueError(f"Sizes at {mode}th mode are expected to be same")
    output_size = (torch.tensor(input.size()) + torch.tensor(other.size()))
    if mode is not None:
        output_size[mode] //= 2
    base = input.new_zeros(output_size.tolist())
    new_input_slice = [slice(0, k) for k in input.size()]
    # slice(start=k, stop=None) is [k, -1]
    new_other_slice = [slice(k, None) for k in input.size()]
    if mode is not None:
        new_other_slice[mode] = slice(0, None)
    base[new_input_slice] = input
    base[new_other_slice] = other
    return base
