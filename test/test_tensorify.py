def test1():
    from tensorify import torch

    input = torch.randn(4, 3, 2)
    other = torch.randn(4, 3)
    assert input.product(other, 1).size() == torch.Size([4, 4, 2])


def test2():
    import torch
    from tensorify import register_as_methods

    register_as_methods()

    input = torch.randn(4, 3, 2)
    other = torch.randn(4, 3)
    assert input.product(other, 1).size() == torch.Size([4, 4, 2])
