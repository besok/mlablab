import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def relu(x):
    return np.maximum(0, x)


def relu_deriv(x):
    return (x > 0).astype(float)


def softmax(x):
    expr = np.exp(x - x.max(axis=1, keepdims=True))
    return expr / expr.sum(axis=1, keepdims=True)


def main():
    """Build and train a small single-hidden-layer network on the iris dataset."""
    iris = load_iris()
    X = iris.data
    y = iris.target

    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42
    )

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    print("Train:", X_train.shape, "Test:", X_test.shape)

    np.random.seed(42)

    W1 = np.random.randn(4, 8) * 0.1  # input -> hidden
    b1 = np.zeros(8)
    W2 = np.random.randn(8, 3) * 0.1  # hidden -> output
    b2 = np.zeros(3)

    def forward(X):
        z1 = X @ W1 + b1
        a1 = relu(z1)
        z2 = a1 @ W2 + b2
        a2 = softmax(z2)
        return z1, a1, z2, a2

    def cross_entropy(pred, target):
        pred = np.clip(pred, 1e-12, 1.0)
        return -np.mean(np.sum(target * np.log(pred), axis=1))

    def backward(X, z1, a1, a2, target):
        m = X.shape[0]
        dz2 = (a2 - target) / m
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ W2.T
        dz1 = da1 * relu_deriv(z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        return dW1, db1, dW2, db2

    learning_rate = 0.1
    epochs = 1000

    for epoch in range(epochs):
        z1, a1, z2, a2 = forward(X_train)
        loss = cross_entropy(a2, y_train)
        dW1, db1, dW2, db2 = backward(X_train, z1, a1, a2, y_train)
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f}")

    _, _, _, preds = forward(X_test)
    predicted_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    accuracy = np.mean(predicted_classes == true_classes)
    print(f"Test accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
