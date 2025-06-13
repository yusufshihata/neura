from __future__ import annotations
import numpy as np
from typing import Optional, Union, List


class Tensor:
    def __init__(
        self,
        data: Union[List[float], np.ndarray],
        requires_grad: bool = True,
        dtype: type = np.float32,
        grad: Optional[np.ndarray] = None,
    ):
        if isinstance(data, list):
            self.data = np.array(data)
        else:
            self.data = data
        self.requires_grad = requires_grad
        self.dtype = dtype
        self.grad = grad
        self.ndim = self.data.ndim
        self.shape = self.data.shape

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

    def __iadd__(self, other: Tensor) -> Tensor:
        self.data += other.data

        return self

    def __isub__(self, other: Tensor) -> Tensor:
        self.data -= other.data

        return self

    def __imul__(self, scalar: np.float32) -> Tensor:
        self.data *= scalar

        return self

    def __imatmul__(self, other: Tensor) -> Tensor:
        self.data @= other.data

        return self

    def __getitem__(self, idx: Union[int, slice]) -> Tensor.dtype:
        return self.data[idx]

    def __len__(self) -> int:
        return self.data.size
    
    def view(self, *args: int) -> Tensor:
        self.data = self.data.reshape(*args)

        return self

    def squeeze(self, dim: int) -> Tensor:
        self.data = self.data.squeeze(axis=dim)

        return self

    def unsqueeze(self, dim: int) -> Tensor:
        self.data = np.expand_dims(self.data, axis=dim)

        return self
    
    def __repr__(self) -> str:
        return f"{self.data}, dtype={self.dtype}"
