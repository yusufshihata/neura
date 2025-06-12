from __future__ import annotations
import numpy as np
from typing import Optional, Union


class Tensor:
    def __init__(
        self,
        data: np.ndarray,
        requires_grad: bool = True,
        dtype: type = np.float32,
        grad: Optional[np.ndarray] = None,
    ):
        self.data = data
        self.requires_grad = requires_grad
        self.dtype = dtype
        self.grad = grad

    def __add__(self, other: Tensor) -> Tensor:
        new_data = self.data + other.data
        requires_grad = self.requires_grad or other.requires_grad
        dtype = self.dtype

        return Tensor(new_data, requires_grad, dtype)

    def __sub__(self, other: Tensor) -> Tensor:
        new_data = self.data - other.data
        requires_grad = self.requires_grad or other.requires_grad
        dtype = self.dtype

        return Tensor(new_data, requires_grad, dtype)

    def __mul__(self, scalar: np.float32) -> Tensor:
        new_data = self.data * scalar

        return Tensor(new_data, self.requires_grad, self.dtype)

    def __matmul__(self, other: Tensor) -> Tensor:
        new_data = np.matmul(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        dtype = self.dtype

        return Tensor(new_data, requires_grad, dtype)

    def __getitem__(self, idx: Union[int, slice]) -> Tensor.dtype:
        return self.data[idx]
    
    def __repr__(self):
        return f"{self.data}, dtype={self.dtype}"

tensor1 = Tensor(np.array([[[2,3,4], [23,34,54], [23, 433, 435]]]))
tensor2 = Tensor(np.array([[[3,46,64], [23,4,54], [23,43,43]]]))

tensor3 = tensor1 + tensor2
tensor4 = tensor1 @ tensor2
tensor5 = tensor1 * 3
tensor6 = tensor1 - tensor2

print(tensor1)
print(tensor2)
print(tensor3)
print(tensor4)
print(tensor5)
print(tensor6)
print(tensor1[0, 2, 2])
print(tensor1[0, 1, 0:2])