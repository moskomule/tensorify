# tensorify [![CircleCI](https://circleci.com/gh/moskomule/tensorify/tree/master.svg?style=svg)](https://circleci.com/gh/moskomule/tensorify/tree/master)

Status is **WIP**.

`tensorify` extends PyTorch tensors to tensor operations.


## Requirements

* PyTorch >= 1.0

## Installation

`pip install git+https://github.com/moskomule/tensorify`

## Usage

```python
import torch
from tensorify import operations

input = torch.randn(4, 3, 2)
other = torch.randn(4, 3)

operations.product(input, other, 1)
# or
input.product(other, 1)

```

Once `tensorify` is called, operations are automatically registered as `torch.Tensor`'s methods.
Or, you can explicity call `tensorify.register_as_methods()`.