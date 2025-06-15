from __future__ import annotations
import numpy as np

class Node:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._children = set(_children)
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
        out = Node(self.data * scalar, (self,), '*')

        def _backward():
            self.grad += scalar * out.grad

        out._backward = _backward

        return out
    
    def __matmul__(self, other: Node) -> Node:
        out = Node(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += other.data.T @ out.grad
            other.grad += self.data @ out.grad.T
        
        out._backward = _backward

        return out
    
    def backward(self):
        graph = []
        visited = set()
        def build_graph(node):
            visited.add(node)
            for child in node._children:
                build_graph(child)
            graph.append(node)
        build_graph(self)
        self.grad = np.ones_like(self.data.shape)
        for node in reversed(graph):
            node._backward()
    
    def zero_grad(self):
        self.grad = 0
        for child in self._children:
            child.zero_grad()
