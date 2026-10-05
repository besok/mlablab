from collections import Counter

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def main():
    """Read handwritten digits without a neural network: k-nearest neighbours on the digits dataset."""
    digits = load_digits()
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

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


if __name__ == "__main__":
    main()
