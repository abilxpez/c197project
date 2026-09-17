import numpy as np
import matplotlib.pyplot as plt

# Load PCA-reduced embeddings and metadata
data = np.load("pca_embeddings.npz", allow_pickle=True)

doc_names = data["doc_names"]
dates = data["dates"]
pca_2d = data["pca_2d"]

# Extract year from each date string
years = np.array([int(date.split("-")[0]) for date in dates])

# Plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=years, cmap='plasma', s=40)

plt.colorbar(scatter, label="Year")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.title("PCA of Presidential Speeches Colored by Year")
plt.grid(True)
plt.tight_layout()
plt.show()
