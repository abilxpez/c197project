import numpy as np

data = np.load("umap_embeddings.npz", allow_pickle=True)

# Show available arrays
print("Keys:", list(data.keys()))

# Peek at shapes and example values
print("doc_names shape:", data["doc_names"].shape)
print("dates shape:", data["dates"].shape)
print("umap_2d shape:", data["umap_2d"].shape)

# Peek at first few rows
print("First doc name:", data["doc_names"][0])
print("First UMAP vector:", data["umap_2d"][0])
