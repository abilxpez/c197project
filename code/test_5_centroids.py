import numpy as np

# Load the saved centroid file
centroids = np.load("label_centroids_normalized.npz")

# List all labels
labels = centroids.files

# Print first 50 (or fewer)
for label in labels[:50]:
    vector = centroids[label]
    print(f"Label: {label}")
    print(f"Centroid shape: {vector.shape}")
    print(f"First 5 values: {vector[:5]}")
    print()
