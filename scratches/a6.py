from collections.abc import Callable

type Forward = Callable[[list[float]], float]
type Backward = Callable[[Node, list[Node]], None]


class Node:
    def __init__(
        self, inputs: list[Node], forward: Forward, backward: Backward
    ) -> None:
        self._inputs = inputs
        self._data = forward([i.data for i in inputs])
        self._grad = 0.0
        self._backward = lambda: backward(self, self._inputs)

    def incr_grad(self, delta: float) -> None:
        self._grad += delta

    @property
    def data(self) -> float:
        return self._data

    @property
    def grad(self) -> float:
        return self._grad

    @property
    def inputs(self) -> list[Node]:
        return self._inputs

    def backward(self) -> None:
        self._backward()

    def __repr__(self) -> str:
        return f"Node(data={self._data:.4f}, grad={self._grad:.4f}, inputs={len(self._inputs)})"

    @classmethod
    def leaf(cls, data: float) -> Node:
        return Node([], lambda _ls: data, lambda _s1, _s2: None)

    def reset_grad(self) -> None:
        visited = set()

        def clear(n: Node) -> None:
            if n not in visited:
                visited.add(n)
                n._grad = 0.0
                for p in n.inputs:
                    clear(p)

        clear(self)

    def traverse_backward(self) -> None:
        topo = []
        visited = set()

        def traverse(n: Node) -> None:
            if n not in visited:
                visited.add(n)
                for p in n.inputs:
                    traverse(p)

                topo.append(n)

        traverse(self)
        self._grad = 1.0
        self.reset_grad()
        for n in reversed(topo):
            n.backward()


def mul(a: Node, b: Node) -> Node:
    return Node(
        [a, b],
        lambda ins: ins[0] * ins[1],
        lambda n, ins: (  # pyright: ignore[reportArgumentType]
            ins[0].incr_grad(ins[1].data * n.grad),
            ins[1].incr_grad(ins[0].data * n.grad),
        ),
    )


# Add ops (each is 5 lines)
#  - add, sub, neg, div, pow(x, n), exp, log, tanh, sigmoid, relu, sum, mean
#
# Prove correctness
#  - Write gradcheck(fn, x_val) — finite differences, ε=1e-6
#  - Run it on 10–20 random expressions composed from your ops
#  - Test deep chain (10+ ops) to stress topo order
#  - Test diamond (a=x*2; b=x*3; y=a+b) → x.grad == 5
#  - Test nested fan-out (x*x + x*x*x) → x.grad = 2x + 3x²
#
# Make it usable
#  - Gradient descent loop on f(x) = (x-3)² — confirm convergence to 3.0
#  - Optimizer as a function — sgd(params, lr) that does p._data -= lr * p._grad
#
# Jump
#  - Port float → np.ndarray
#  - Handle broadcasting in backward — every op's backward must sum over broadcasted axes
#  - matmul with backward: dA = dOut @ B.T, dB = A.T @ dOut
#  - reshape, transpose
#  - Re-run gradcheck with arrays — this catches every broadcasting bug
#  - Linear layer — W @ x + b as a class or function
#  - Loss functions — MSE, then cross-entropy
#  - Train on MNIST
#
def main() -> None:
    x = Node.leaf(3.0)
    y = mul(x, x)
    y.traverse_backward()
    assert x.grad == 6.0


if __name__ == "__main__":
    main()
