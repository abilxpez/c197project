import numpy as np
from sklearn.decomposition import PCA

# Load the original high-dimensional embeddings
data = np.load("speech_embeddings.npz", allow_pickle=True)

doc_names = data["doc_names"]
dates = data["dates"]
embeddings = data["embeddings"]

# Run PCA to reduce from 1536 → 2 dimensions
print("Running PCA...")
pca = PCA(n_components=2)
pca_embeddings = pca.fit_transform(embeddings)

print(f"PCA completed. Shape: {pca_embeddings.shape}")
print(f"Explained variance: {pca.explained_variance_ratio_}")

# Save the 2D PCA results + metadata
np.savez("pca_embeddings.npz",
         doc_names=doc_names,
         dates=dates,
         pca_2d=pca_embeddings)

print("Saved PCA results to pca_embeddings.npz")
