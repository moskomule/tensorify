import pytest
import torch

from tensorify import operations


def test_matricization():
    input = torch.randn(4, 3, 2, 5, requires_grad=True)
    matrix = operations.matricization(input, 2)
    assert matrix.size() == torch.Size([2, 60])
    # check if backward-able
    matrix.sum().backward()


def test_canonical_matricization():
    input = torch.randn(4, 3, 2, 5, requires_grad=True)
    matrix = operations.canonical_matricization(input, 2)
    assert matrix.size() == torch.Size([24, 5])
    # check if backward-able
    matrix.sum().backward()


def test_product():
    input = torch.randn(4, 3, 2, requires_grad=True)
    other = torch.randn(4, 3)
    product = operations.product(input, other, 1)
    assert product.size() == torch.Size([4, 4, 2])
    # check if backward-able
    product.sum().backward()

    other = torch.randn(3)
    assert operations.product(input, other, 1).size() == torch.Size([4, 2])

    with pytest.raises(ValueError):
        operations.product(input, other, 2)

    with pytest.raises(ValueError):
        # other is expected to be vector or matrix but got tensor
        operations.product(input, input, 1)


def test_multilinier_product():
    core = torch.randn(4, 3, 2, requires_grad=True)
    factors = [torch.randn(3, 4, requires_grad=True),
               torch.randn(3, 3, requires_grad=True),
               torch.randn(3, 2, requires_grad=True)]
    product = operations.multilinier_product(core, factors)
    assert product.size() == torch.Size([3, 3, 3])
    # check if backward-able
    product.sum().backward()

    factors.pop()
    with pytest.raises(ValueError):
        operations.multilinier_product(core, factors)


def test_contracted_product():
    input = torch.randn(4, 3, 2, requires_grad=True)
    other = torch.randn(2, 3, 4)
    product = operations.contracted_product(input, other, None)
    assert product.size() == torch.Size([4, 3, 3, 4])
    # check if backward-able
    product.sum().backward()
    with pytest.raises(ValueError):
        operations.contracted_product(input, torch.randn(3, 4, 2), modes=(2, 0))


def test_outer_prodcut():
    input = torch.randn(4, 3, 2, requires_grad=True)
    other = torch.randn(2, 3, 4)
    product = operations.outer_product(input, other)
    assert product.size() == torch.Size([4, 3, 2, 2, 3, 4])
    # check if backward-able
    product.sum().backward()


def test_kronecker_product():
    input = torch.randn(4, 3, 2)
    other = torch.randn(3, 3, 3)
    input[0, 0, 0] = 1
    input.requires_grad_()
    product = operations.kronecker_product(input, other)
    # size check
    assert product.size() == torch.Size([4 * 3, 3 * 3, 2 * 3])
    # value check
    assert torch.equal(product[0:3, 0:3, 0:3], other)
    # check if backward-able
    product.sum().backward()


def test_direct_sum():
    input = torch.randn(3, 3, 3)
    other = torch.randn(4, 4, 3)
    result = operations.direct_sum(input, other)
    assert result.size() == torch.Size([7, 7, 6])

    result = operations.direct_sum(input, other, 2)
    assert result.size() == torch.Size([7, 7, 3])
