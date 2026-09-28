# /// script
# dependencies = [
#     "marimo>=0.25.0",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fit a line to height (cm) vs. weight (kg) points using a simple gradient descent loop.

    $ Weight = w*Height + b  $

    The Outer Function (Squared Error): $J(e) = \frac{1}{N} \sum e^2$
    The Inner Function (The Error): $e = (w x + b) - y$

    The final gradient:
    $\frac{\partial J}{\partial b} = \frac{2}{N} \sum (e \cdot 1)$

    Statistical scaling is needed due to imbalance for the height (too big values).
    Possible to translate it to meters but more robust to apply scaling $z = \frac{x - \mu}{\sigma}$
    where:
    $x$ is the raw input value.$\mu$ is the mean of the dataset.$\sigma$ is the standard deviation of the dataset.
    """)
    return


@app.cell
def _():
    import numpy as np

    heights = np.round(np.random.normal(loc=170, scale=10, size=10, ))
    noise = np.random.normal(loc=0, scale=3, size=10)
    weights = np.round(0.6 * heights - 10 + noise)

    X = heights
    Y = weights
    X_scaled = (X - X.mean()) / X.std()
    w, b = 0., 0.
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
    return


if __name__ == "__main__":
    app.run()
