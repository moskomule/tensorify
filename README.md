# tensorify [![CircleCI](https://circleci.com/gh/moskomule/tensorify/tree/master.svg?style=svg)](https://circleci.com/gh/moskomule/tensorify/tree/master)

Status is **WIP**.

`tensorify` extends PyTorch tensors to tensor operations.


## Requirements

`tensorify` is pure PyTorch, without any other dependenccies.

* Python >= 3.7
* PyTorch >= 1.0

## Installation

`pip install git+https://github.com/moskomule/tensorify`

## Usage

Some basic operations are supported. `tensorify` adds such operations to `torch.Tensor`.

```python
import torch
# or `from tensorify import torch`
from tensorify import operations

input = torch.randn(4, 3, 2)
other = torch.randn(4, 3)

operations.product(input, other, 1)
# or
input.product(other, 1)
```

Also, some tensor decomposition methods are supported. Some operations between decomposed tensors can be done without re-composing.

```python
from tensorify.decompositions import TensorTrain

tt1 = TensorTrain()
tt2 = TensorTrain()
tt1.decompose(input)
tt2.decompose(input)
# decomposed tensor supports some operations
tt1 + tt2
```

Once `tensorify` is called, operations are automatically registered as `torch.Tensor`'s methods.
Or, you can explicity call `tensorify.register_as_methods()`.

## Related Project

* [Tensorly](https://github.com/tensorly/tensorly)