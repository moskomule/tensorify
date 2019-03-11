import pytest
import torch

from tensorify import operations


def test_product():
    input = torch.randn(4, 3, 2)
    other = torch.randn(4, 3)
    assert operations.product(input, other, 1).size() == torch.Size([4, 4, 2])

    other = torch.randn(3)
    assert operations.product(input, other, 1).size() == torch.Size([4, 2])

    with pytest.raises(ValueError):
        operations.product(input, other, 2)

    with pytest.raises(ValueError):
        # other is expected to be vector or matrix but got tensor
        operations.product(input, input, 1)


def test_multilinier_product():
    core = torch.randn(4, 3, 2)
    factors = [torch.randn(3, 4),
               torch.randn(3, 3),
               torch.randn(3, 2)]
    assert operations.multilinier_product(core, factors).size() == torch.Size([3, 3, 3])
    factors.pop()
    with pytest.raises(ValueError):
        operations.multilinier_product(core, factors)


def test_contracted_product():
    input = torch.randn(4, 3, 2)
    other = torch.randn(2, 3, 4)
    assert operations.contracted_product(input, other).size() == torch.Size([4, 3, 3, 4])
    with pytest.raises(ValueError):
        operations.contracted_product(input, torch.randn(3, 4, 2))


def test_outer_prodcut():
    input = torch.randn(4, 3, 2)
    other = torch.randn(2, 3, 4)
    assert operations.outer_product(input, other).size() == torch.Size([4, 3, 2, 2, 3, 4])
