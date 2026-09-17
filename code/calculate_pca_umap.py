import json
import numpy as np
from sklearn.decomposition import PCA
import umap

# Load the JSON data
with open("merged_labeled_speeches.json", "r") as f:
    data = json.load(f)

# Extract embeddings
embeddings = [entry["embedding"] for entry in data]

# Compute PCA (2D)
pca_model = PCA(n_components=2)
pca_result = pca_model.fit_transform(embeddings)

# Compute UMAP (2D)
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_result = umap_model.fit_transform(embeddings)

# Add PCA and UMAP results to each entry
for i, entry in enumerate(data):
    entry["pca"] = pca_result[i].tolist()
    entry["umap"] = umap_result[i].tolist()

# Save the updated data
output_path = "merged_labeled_speeches_umap_pca.json"
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

output_path
