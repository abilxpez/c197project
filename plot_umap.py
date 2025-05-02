import numpy as np
import matplotlib.pyplot as plt

# Load UMAP results
data = np.load("umap_embeddings.npz", allow_pickle=True)

doc_names = data["doc_names"]
dates = data["dates"]
umap_2d = data["umap_2d"]

# Convert year from date string
years = np.array([int(date.split("-")[0]) for date in dates])

# Plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(umap_2d[:, 0], umap_2d[:, 1], c=years, cmap='plasma', s=40)

plt.colorbar(scatter, label="Year")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title("UMAP of Presidential Speeches Colored by Year")
plt.grid(True)
plt.tight_layout()
plt.show()
