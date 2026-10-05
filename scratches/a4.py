from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def main():
    """Use k-means to extract the dominant colours from your own photos."""
    image_path = str(Path(__file__).parent / "assets" / "456626.jpg")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape((-1, 3))

    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    counts = np.bincount(labels)
    percentages = counts / len(pixels)

    sorted_indices = np.argsort(percentages)[::-1]
    dominant_colors = dominant_colors[sorted_indices]
    percentages = percentages[sorted_indices]

    palette_bar = np.zeros((50, 300, 3), dtype="uint8")
    start_x = 0

    for percent, color in zip(percentages, dominant_colors):
        end_x = start_x + int(percent * 300)
        cv2.rectangle(palette_bar, (int(start_x), 0), (int(end_x), 50), color.tolist(), -1)
        start_x = end_x

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [4, 1]})

    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(palette_bar)
    ax2.set_title(f"Extracted {k}-Color Palette")
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
