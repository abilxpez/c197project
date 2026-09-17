import json
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = "merged_labeled_speeches_umap_pca.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Normalize party labels
def normalize_party(party):
    if party in ["D", "Democrat"]:
        return "Democrat"
    elif party in ["R", "Republican"]:
        return "Republican"
    return "Other"

# Prepare UMAP, PCA, and party data
umap_coords = []
pca_coords = []
party_labels = []

for entry in data:
    party = normalize_party(entry.get("speaker_party", "Unknown"))
    if party != "Other":  # Exclude unknown/other
        umap_coords.append(entry["umap"])
        pca_coords.append(entry["pca"])
        party_labels.append(party)

umap_coords = np.array(umap_coords)
pca_coords = np.array(pca_coords)

# Map parties to colors
party_to_color = {"Democrat": "blue", "Republican": "red"}
colors = [party_to_color[p] for p in party_labels]

# Plot UMAP
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(umap_coords[:, 0], umap_coords[:, 1], c=colors, s=10, alpha=0.6)
plt.title("UMAP Colored by Party")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.grid(True)

# Plot PCA
plt.subplot(1, 2, 2)
plt.scatter(pca_coords[:, 0], pca_coords[:, 1], c=colors, s=10, alpha=0.6)
plt.title("PCA Colored by Party")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)

# Add legend
legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', label='Democrat',
               markerfacecolor='blue', markersize=6),
    plt.Line2D([0], [0], marker='o', color='w', label='Republican',
               markerfacecolor='red', markersize=6)
]
plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', title="Party")

plt.tight_layout()
plt.show()
