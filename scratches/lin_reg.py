import numpy as np


def main() -> None:
    X1 = np.random.normal(0, 1, 100)
    X2 = X1 + np.random.normal(0, 1e-6, 100)

    X = np.column_stack([X1, X2])

    true_w = np.array([4.0, 5.0])

    y = X @ true_w + np.random.normal(0, 0.1, 100)

    real_w = np.linalg.inv(X.T @ X + np.eye(2)) @ X.T @ y

    print(real_w)


if __name__ == "__main__":
    main()
