from typing import Callable
import numpy as np


class Node:
    def __init__(
        self,
        inputs: list[Node],
        forward: Callable[[list[Node]], np.ndarray],
        backward: Callable[[Node, list[Node]], None],
        data: np.ndarray | None = None,
    ) -> None:
        self._inputs = inputs
        self._forward = forward
        self.backward = backward
        self._data = data if data is not None else forward(inputs)
        self._grad = np.zeros_like(self._data, dtype=np.float64)

    @classmethod
    def leaf(cls, data: np.ndarray) -> Node:
        arr = np.asarray(data, dtype=np.float64)
        node = cls(
            inputs=[],
            forward=lambda _ins: node._data,
            backward=lambda n1, n2: None,
            data=arr,
        )
        return node

    def forward(self) -> np.ndarray:
        for input in self.inputs:
            input.forward()

        self._data = self._forward(self.inputs)

        if self.grad is None or self._grad.shape != self._data.shape:
            self._grad = np.zeros_like(self._data, dtype=np.float64)

        return self.data

    def reset_grad(self) -> None:
        visited = set()

        def clear(n: Node) -> None:
            if n not in visited:
                visited.add(n)
                n._grad = np.zeros_like(n.data, dtype=np.float64)
                for p in n.inputs:
                    clear(p)

        clear(self)

    def traverse_backward(self) -> None:
        self.reset_grad()
        topo = []
        visited = set()

        def traverse(n: Node) -> None:
            if n not in visited:
                visited.add(n)
                for p in n.inputs:
                    traverse(p)
                topo.append(n)

        traverse(self)
        self._grad = np.ones_like(self._data, dtype=np.float64)
        for n in reversed(topo):
            n.backward(n, n.inputs)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def grad(self) -> np.ndarray:
        return self._grad

    @property
    def inputs(self) -> list[Node]:
        return self._inputs
