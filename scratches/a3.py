import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read handwritten digits without a neural network
    Use k-nearest neighbours on scikit-learn's small digits dataset.
    """)
    return


@app.cell
def _():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return (
        KNeighborsClassifier,
        X_test,
        X_train,
        accuracy_score,
        y_test,
        y_train,
    )


@app.cell
def _(KNeighborsClassifier, X_test, X_train, accuracy_score, y_test, y_train):
    def _():
        knn = KNeighborsClassifier(n_neighbors=5)

        knn.fit(X_train, y_train)

        preds = knn.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        return print(f"Model Accuracy: {accuracy * 100:.2f}%")


    _()
    return


@app.cell
def _(KNeighborsClassifier, X_test, X_train, y_test, y_train):
    from collections import Counter

    knn = KNeighborsClassifier(n_neighbors=5)

    knn.fit(X_train, y_train)

    test_index = 0
    single_test_point = X_test[test_index].reshape(1, -1) 
    actual_label = y_test[test_index]

    print(f"Testing image at test_index {test_index}. Actual True Label: {actual_label}")
    print("-" * 50)

    distances, neighbor_indices = knn.kneighbors(single_test_point)
    nearest_indices = neighbor_indices[0] 
    nearest_distances = distances[0]
    neighbor_labels = y_train[nearest_indices]

    for i in range(5):
        idx = nearest_indices[i]
        lbl = neighbor_labels[i]
        dist = nearest_distances[i]
        print(f"Neighbor {i+1} -> Training Index: {idx:<4} | Label: {lbl} | Distance: {dist:.2f}")
    print("-" * 50)

    vote_counts = Counter(neighbor_labels)
    final_prediction = knn.predict(single_test_point)[0]

    print(f"The Vote: {dict(vote_counts)}")
    print(f"Final Algorithm Prediction: {final_prediction}")

    return


if __name__ == "__main__":
    app.run()
