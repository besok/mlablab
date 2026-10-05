import numpy as np


def main():
    """Fit a line to height (cm) vs. weight (kg) points using a simple gradient descent loop.

    Weight = w*Height + b

    Outer function (squared error): J(e) = (1/N) * sum(e^2)
    Inner function (the error):    e = (w*x + b) - y

    Final gradient:
        dJ/db = (2/N) * sum(e * 1)

    Statistical scaling is needed due to imbalance for the height (too big values).
    """
    heights = np.round(np.random.normal(loc=170, scale=10, size=10))
    noise = np.random.normal(loc=0, scale=3, size=10)
    weights = np.round(0.6 * heights - 10 + noise)

    X = heights
    Y = weights
    X_scaled = (X - X.mean()) / X.std()
    w, b = 0.0, 0.0
    alpha = 0.1

    for _ in range(100):
        Y_pred = X_scaled * w + b
        error = Y_pred - Y
        dw = (2 / len(X)) * np.dot(error, X_scaled)
        db = (2 / len(X)) * np.sum(error)
        w = w - alpha * dw
        b = b - alpha * db

    print(f"Fitted model: Weight = {w:.2f} * (Scaled Height) + {b:.2f}")
    for x, y in zip(X_scaled, Y):
        print(f" {(w * x + b):.1f} == {y}")


if __name__ == "__main__":
    main()
