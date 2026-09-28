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
    ### Predict from a table
     Classify penguin species from body measurements with a decision tree. Then deliberately overfit it.
    """)
    return


@app.cell
def _():
    import seaborn as sns

    from sklearn.model_selection import train_test_split

    penguins = sns.load_dataset('penguins')
    penguins.dropna()

    features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    X = penguins[features]
    y = penguins["species"]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_test, X_train, y_test, y_train):
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    import matplotlib.pyplot as plt
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)

    clf.fit(X_train, y_train)

    print(f"Accuracy: {clf.score(X_test, y_test):.2%}")
    return (DecisionTreeClassifier,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Overfitting**
    """)
    return


@app.cell
def overfitting(DecisionTreeClassifier, X_test, X_train, y_test, y_train):
    import numpy as np
    y_train_noisy = y_train.copy()
    noise_mask = np.random.rand(len(y_train)) < 0.25
    y_train_noisy[noise_mask] = np.random.choice(y_train.unique(), size=noise_mask.sum())

    clf_over = DecisionTreeClassifier(
        max_depth=None,
        random_state=42)

    clf_over.fit(X_train, y_train_noisy)

    print(f"Accuracy: {clf_over.score(X_test, y_test):.2%}")
    return


if __name__ == "__main__":
    app.run()
