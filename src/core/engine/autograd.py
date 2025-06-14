from __future__ import annotations
import numpy as np

class Node:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __add__(self, other: Node) -> Node:
        out = Node(self.data + other.data, (self, other), '+')


        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward

        return out
    
    def __sub__(self, other: Node) -> Node:
        out = Node(self.data - other.data, (self, other), '-')


        def _backward():
            self.grad += 1 * out.grad
            other.grad -= 1 * out.grad

        out._backward = _backward

        return out
    
    def __mul__(self, scalar: Node) -> Node:
        out = Node(self.data * scalar, (self, scalar), '*')

        def _backward():
            self.grad += scalar * out.grad
            scalar.grad += self.data * out.grad

        out._backward = _backward

        return out
    
    def __matmul__(self, other: Node) -> Node:
        out = Node(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += other.data.T @ out.grad
            other.grad += self.data @ out.grad.T
        
        out._backward = _backward

        return out
