import numpy as np
import umap
import os

# Load the embeddings and metadata
data = np.load("speech_embeddings.npz", allow_pickle=True)

doc_names = data["doc_names"]
dates = data["dates"]
embeddings = data["embeddings"]

# Run UMAP to reduce from 1536 → 2 dimensions
print("Running UMAP...")
reducer = umap.UMAP(n_components=2, random_state=42)
umap_embeddings = reducer.fit_transform(embeddings)

print(f"UMAP completed. Shape: {umap_embeddings.shape}")

# Save the 2D UMAP results + metadata
np.savez("umap_embeddings.npz",
         doc_names=doc_names,
         dates=dates,
         umap_2d=umap_embeddings)

print("Saved UMAP results to umap_embeddings.npz")
