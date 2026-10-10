import builtins
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

    def set_data(self, v: float) -> None:
        self._data = v

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
        self._grad = 1.0
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


def add(a: Node, b: Node) -> Node:
    return Node(
        [a, b],
        lambda ins: ins[0] + ins[1],
        lambda n, ins: (
            ins[0].incr_grad(n.grad),
            ins[1].incr_grad(n.grad),
        ),
    )


def sub(a: Node, b: Node) -> Node:
    return Node(
        [a, b],
        lambda ins: ins[0] - ins[1],
        lambda n, ins: (
            ins[0].incr_grad(n.grad),
            ins[1].incr_grad(-n.grad),
        ),
    )


def div(a: Node, b: Node) -> Node:
    return Node(
        [a, b],
        lambda ins: ins[0] / ins[1],
        lambda n, ins: (
            ins[0].incr_grad(n.grad / ins[1].data),
            ins[1].incr_grad(-1 * n.grad * ins[0].data / ins[1].data ** 2),
        ),
    )


def neg(a: Node) -> Node:
    return Node(
        [a],
        lambda ins: -ins[0],
        lambda n, ins: ins[0].incr_grad(-1 * n.grad),
    )


def pow(a: Node, n: int) -> Node:
    return Node(
        [a],
        lambda ins: ins[0] ** n,
        lambda node, ins: ins[0].incr_grad(n * ins[0].data ** (n - 1) * node.grad),
    )


def exp(a: Node) -> Node:
    return Node(
        [a],
        lambda ins: -ins[0],
        lambda n, ins: ins[0].incr_grad(-1 * n.grad),
    )


def relu(a: Node) -> Node:
    return Node(
        [a],
        lambda ins: max(0.0, ins[0]),
        lambda n, ins: ins[0].incr_grad(n.grad if ins[0].data > 0.0 else 0.0),
    )


def sum(nodes: list[Node]) -> Node:

    def _backward(n: Node, inputs: list[Node]) -> None:
        for i in inputs:
            i.incr_grad(n.grad)

    return Node(nodes, lambda ins: builtins.sum(ins), _backward)


def mean(nodes: list[Node]) -> Node:

    def _backward(n: Node, inputs: list[Node]) -> None:
        for i in inputs:
            i.incr_grad(n.grad / len(inputs))

    return Node(nodes, lambda ins: builtins.sum(ins) / len(ins), _backward)


def gradcheck(
    f: Callable[[list[Node]], Node],
    inputs: list[float],
    eps: float = 1e-6,
    tol: float = 1e-5,
) -> bool:
    nodes = [Node.leaf(i) for i in inputs]
    out = f(nodes)
    out.traverse_backward()

    analytical_grads = [n.grad for n in nodes]
    numerical_grads = []

    for i in range(len(inputs)):
        i_inputs = list(inputs)
        i_inputs[i] += eps

        out_plus = f([Node.leaf(i) for i in i_inputs]).data
        i_inputs[i] -= eps * 2
        out_minus = f([Node.leaf(i) for i in i_inputs]).data
        numerical_grads.append((out_plus - out_minus) / (2 * eps))

    all_passed = True
    print(
        f"{'Input i':<8} | {'Analytical':<15} | {'Numerical':<15} | {'Rel Error':<12}"
    )
    print("-" * 58)

    for i, (a, n) in enumerate(zip(analytical_grads, numerical_grads)):
        rel_err = abs(n - a) / (max(1e-8, abs(a) + abs(n)))
        passed = rel_err < tol
        if not passed:
            all_passed = False
        status = "OK" if passed else "FAIL"
        print(f"{i:<8} | {a:<15.8f} | {n:<15.8f} | {rel_err:<12.2e} [{status}]")

    return all_passed


def sgd(params: list[Node], lr: float) -> None:
    for p in params:
        p.set_data(p.data - lr * p.grad)


def grad_descent(
    expr: Callable[[list[Node]], Node],
    params: list[Node],
    optimizer: Callable[[list[Node], float], None] = sgd,
    learning_rate: float = 0.1,
    attempts: int = 100,
    tol: float = 1e-6,
) -> list[Node]:

    for _ in range(attempts):
        f = expr(params)
        f.traverse_backward()

        if builtins.sum(abs(p.grad) for p in params) < tol:
            break

        optimizer(params, learning_rate)

    return params


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
    params = [Node.leaf(0.0), Node.leaf(0.0)]

    expr = lambda ns: add(
        pow(sub(ns[0], Node.leaf(3.0)), 2), pow(add(ns[1], Node.leaf(2.0)), 2)
    )

    result = grad_descent(expr, params)
    print(f"x: {result[0].data:.4f} (target: 3.0)")
    print(f"y: {result[1].data:.4f} (target: -2.0)")


if __name__ == "__main__":
    main()
