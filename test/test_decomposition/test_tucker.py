import torch

from tensorify import decompositions


def test_tucker():
    input = torch.randn(4, 5, 6)
    tucker = decompositions.Tucker()
    # core, factors = tucker.decompose(input,
    #                                  rank=3,
    #                                  max_iter=2)
    # composed = tucker.tucker_composition(core, factors)
    # assert input.size() == composed.size()
