import pytest
import torch

from tensorify import decompositions


def test_cp():
    input = torch.randn(4, 5, 6)
    cp = decompositions.CP()
    core, factors = cp.decompose(input,
                                 rank=3,
                                 max_iter=2)
    composed = cp.cp_compose(core, factors)
    assert input.size() == composed.size()

    # should be ok
    decompositions.CP.als(input, 4)
    with pytest.raises(ValueError):
        # should be ng
        decompositions.CP.als(input, 5)
