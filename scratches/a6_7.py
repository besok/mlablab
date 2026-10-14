import builtins
from ast import Pass
from typing import Callable

import numpy as np
from sklearn.datasets import fetch_openml


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

    def set_data(self, v: np.ndarray) -> None:
        self._data = v

    def null_grad(self) -> None:
        self._grad = np.zeros_like(self.data)
    
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

    def incr_grad(self, delta: np.ndarray) -> None:
        self._grad += delta

    @property
    def inputs(self) -> list[Node]:
        return self._inputs


def unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    for a, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=a, keepdims=True)

    return grad


def sum_tensor(a: Node) -> Node:
    def _forward(inputs: list[Node]) -> np.ndarray:
        return np.sum(inputs[0].data)

    def _backward(n: Node, inputs: list[Node]) -> None:
        inp_a = inputs[0]
        inp_a.incr_grad(np.broadcast_to(n.grad, inp_a.data.shape))

    return Node(
        inputs=[a],
        forward=_forward,
        backward=_backward,
    )


def add(a: Node, b: Node) -> Node:
    def _f(inputs: list[Node]) -> np.ndarray:
        return inputs[0].data + inputs[1].data

    def _b(n: Node, inputs: list[Node]) -> None:
        [an, bn] = inputs
        an.incr_grad(unbroadcast(n.grad, an.data.shape))
        bn.incr_grad(unbroadcast(n.grad, bn.data.shape))

    return Node(inputs=[a, b], forward=_f, backward=_b)


def sub(a: Node, b: Node) -> Node:
    def _f(inputs: list[Node]) -> np.ndarray:
        return inputs[0].data - inputs[1].data

    def _b(n: Node, inputs: list[Node]) -> None:
        [an, bn] = inputs
        an.incr_grad(unbroadcast(n.grad, an.data.shape))
        bn.incr_grad(unbroadcast(-n.grad, bn.data.shape))

    return Node(inputs=[a, b], forward=_f, backward=_b)


def mul(a: Node, b: Node) -> Node:
    def _f(inputs: list[Node]) -> np.ndarray:
        return inputs[0].data * inputs[1].data

    def _b(n: Node, inputs: list[Node]) -> None:
        [an, bn] = inputs
        an.incr_grad(unbroadcast(n.grad * bn.data, an.data.shape))
        bn.incr_grad(unbroadcast(n.grad * an.data, bn.data.shape))

    return Node(inputs=[a, b], forward=_f, backward=_b)


def div(a: Node, b: Node) -> Node:
    def _f(inputs: list[Node]) -> np.ndarray:
        return inputs[0].data / inputs[1].data

    def _b(n: Node, inputs: list[Node]) -> None:
        [an, bn] = inputs
        an.incr_grad(unbroadcast(n.grad / bn.data, an.data.shape))
        bn.incr_grad(unbroadcast(-1 * n.grad * an.data / (bn.data**2), bn.data.shape))

    return Node(inputs=[a, b], forward=_f, backward=_b)


def neg(a: Node) -> Node:
    def _f(inputs: list[Node]) -> np.ndarray:
        return -inputs[0].data

    def _b(n: Node, inputs: list[Node]) -> None:
        [an] = inputs
        an.incr_grad(unbroadcast(-1 * n.grad, an.data.shape))

    return Node(inputs=[a], forward=_f, backward=_b)


def pow(a: Node, n: int) -> Node:
    def _f(inputs: list[Node]) -> np.ndarray:
        return inputs[0].data ** n

    def _b(node: Node, inputs: list[Node]) -> None:
        [an] = inputs
        an.incr_grad(unbroadcast(n * (an.data ** (n - 1)) * node.grad, an.data.shape))

    return Node(inputs=[a], forward=_f, backward=_b)


def relu(a: Node) -> Node:
    def _f(inputs: list[Node]) -> np.ndarray:
        return np.maximum(0.0, inputs[0].data)

    def _b(n: Node, inputs: list[Node]) -> None:
        [an] = inputs
        an.incr_grad(unbroadcast(n.grad * (an.data > 0.0), an.data.shape))

    return Node(inputs=[a], forward=_f, backward=_b)


def matmul(a: Node, b: Node) -> Node:
    def _f(inputs: list[Node]) -> np.ndarray:
        return inputs[0].data @ inputs[1].data

    def _b(n: Node, inputs: list[Node]) -> None:
        [an, bn] = inputs
        an.incr_grad(n.grad @ bn.data.T)
        bn.incr_grad(an.data.T @ n.grad)

    return Node(inputs=[a, b], forward=_f, backward=_b)


def gradcheck(
    f: Callable[[list[Node]], Node],
    inputs: list[np.ndarray],
    eps: float = 1e-6,
    tol: float = 1e-5,
) -> bool:
    nodes = [Node.leaf(i) for i in inputs]
    out = f(nodes)
    out.traverse_backward()

    analytical_grads = [n.grad.copy() for n in nodes]
    all_passed = True
    print(
        f"{'Input':<6} | {'Index':<10} | {'Analytical':<12} | {'Numerical':<12} | {'Rel Error':<12} | Status"
    )

    for input_idx, arr in enumerate(inputs):
        num_grad = np.zeros_like(arr, dtype=np.float64)
        for idx in np.ndindex(arr.shape):
            inp_plus = [arr.copy() for arr in inputs]
            inp_plus[input_idx][idx] += eps
            nodes_plus = [Node.leaf(a) for a in inp_plus]
            out_plus = f(nodes_plus).data.item()

            inp_minus = [arr.copy() for arr in inputs]
            inp_minus[input_idx][idx] -= eps
            nodes_minus = [Node.leaf(a) for a in inp_minus]
            out_minus = f(nodes_minus).data.item()

            num_grad[idx] = (out_plus - out_minus) / (2 * eps)
            a_val = analytical_grads[input_idx][idx]
            n_val = num_grad[idx]

            rel_err = abs(n_val - a_val) / max(1e-8, abs(a_val) + abs(n_val))
            passed = rel_err < tol

            if not passed:
                all_passed = False

            status = "OK" if passed else "FAIL"
            print(
                f"{input_idx:<6} | {str(idx):<10} | {a_val:<12.6f} | {n_val:<12.6f} | {rel_err:<12.2e} | {status}"
            )
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
        f.forward()
        f.traverse_backward()

        total_grad_norm = builtins.sum(float(np.sum(np.abs(p.grad))) for p in params)

        if total_grad_norm < tol:
            break

        optimizer(params, learning_rate)

    return params


def mse_loss(y_pred: Node, y_true: Node) -> Node:
    """Mean Squared Error loss: mean((y_pred - y_true)^2)."""
    diff = sub(y_pred, y_true)
    squared_diff = mul(diff, diff)
    loss_sum = sum_tensor(squared_diff)
    num_elements = float(y_pred.data.size)
    scale_node = Node.leaf(np.array([[1.0 / num_elements]], dtype=np.float64))

    return mul(loss_sum, scale_node)


def softmax_cross_entropy(logits: Node, targets: Node) -> Node:

    def _forward(inputs: list[Node]) -> np.ndarray:
        z = inputs[0].data
        y = inputs[1].data
        batch_size = z.shape[0]

        z_shift = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shift)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        loss = (
            -np.sum(y * (z_shift - np.log(np.sum(exp_z, axis=1, keepdims=True))))
            / batch_size
        )
        return np.array([[loss]], dtype=np.float64)

    def _backward(n: Node, inputs: list[Node]) -> None:
        [logits_node, targets_node] = inputs
        z = logits_node.data
        y = targets_node.data
        batch_size = z.shape[0]

        z_shift = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shift)
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        dlogits = (probs - y) / batch_size

        logits_node.incr_grad(unbroadcast(n.grad * dlogits, logits_node.data.shape))

    return Node(inputs=[logits, targets], forward=_forward, backward=_backward)

class Module:
    def forward(self, *args, **kwargs) -> Node:
        raise NotImplemented()    

    def parameters(self) -> list[Node]:
        params = []
        for value in self.__dict__.values():
            if isinstance(value, Node):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
        return params

class Linear(Module):
    def __init__(self, in_fts: int, out_fts:int) -> None:
        bound = 1.0  / np.sqrt(in_fts)

        w_data = np.random.uniform(-bound, bound, (in_fts,out_fts))        
        b_data = np.zeros((1, out_fts),dtype=np.float64)

        self.W = Node.leaf(w_data)
        
        self.b = Node.leaf(b_data)
        super().__init__()

    def forward(self, x: Node) -> Node:
        return add(matmul(x, self.W), self.b)


class MLP(Module):
    def __init__(self, input:int, hidden:int, output:int) -> None:
        self.fc1 = Linear(input, hidden)
        self.fc2 = Linear(hidden, output)
        super().__init__()

    def forward(self, x: Node) -> Node:
        return self.fc2.forward(relu(self.fc1.forward(x)))


def get_mnist_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float64) / 255.0
    y_int = y.astype(np.int64)
    y_one_h = np.zeros((y_int.size, 10),dtype=np.float64)
    y_one_h[np.arange(y_int.size), y_int] = 1.0

    return X[:60000], y_one_h[:60000], X[60000:], y_one_h[60000:]


def evaluate(model:Module, X_test: np.ndarray, y_test:np.ndarray, batch_size:int = 500) -> float:
    correct = 0
    num_samples = X_test.shape[0]

    for start_idx in range(0, num_samples, batch_size):
        x_b = X_test[start_idx: start_idx + batch_size]
        y_b = y_test[start_idx: start_idx + batch_size]

        logits = model.forward(Node.leaf(x_b))
        logits.forward()

        preds = np.argmax(logits.data,axis=1)
        targs = np.argmax(y_b, axis=1)

        correct += np.sum(preds == targs)
        
    return (correct/num_samples) * 100


def main() -> None:
    X_train, y_train, X_test, y_test = get_mnist_data()

    model = MLP(784,50,10)
    params = model.parameters()

    batch_size = 64
    epochs = 3
    learning_rate = 0.1

    num_samples = X_train.shape[0]

    for epoch in range(epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)


        epoch_loss = 0.0
        correct_preds = 0

        for start_idx in range(0, num_samples, batch_size):
            batch_idx = indices[start_idx: start_idx + batch_size]

            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            for p in params:
                p.null_grad()

            x_node = Node.leaf(X_batch)
            y_node = Node.leaf(y_batch)

            logits = model.forward(x_node)
            loss_node = softmax_cross_entropy(logits, y_node)

            loss_node.forward()
            loss_node.traverse_backward()

            sgd(params, learning_rate)
            epoch_loss += loss_node.data.item() * X_batch.shape[0]
            preds = np.argmax(logits.data, axis=1)
            targs = np.argmax(y_batch, axis = 1)
            correct_preds += np.sum(preds == targs)

            
        avg_loss = epoch_loss / num_samples
        acc = (correct_preds / num_samples) * 100

        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%"
        )

        test_acc = evaluate(model, X_test, y_test)
        print(
            f"Test Acc: {test_acc:.2f}%"
        )
        
    
if __name__ == "__main__":
    main()
