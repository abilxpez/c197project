import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.cm import get_cmap

# Load the data
file_path = "merged_labeled_speeches_umap_pca.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Count label frequencies and get top 10
label_counter = Counter()
for entry in data:
    for label in entry.get("labels", []):
        label_counter[label] += 1

top_labels = [label for label, _ in label_counter.most_common(10)]

# Prepare UMAP and PCA data filtered by top labels
umap_coords = []
pca_coords = []
colors = []
labels_used = []

for entry in data:
    entry_labels = entry.get("labels", [])
    if not entry_labels:
        continue

    # Use only the first label that is in top_labels
    label = next((l for l in entry_labels if l in top_labels), None)
    if label:
        umap_coords.append(entry["umap"])
        pca_coords.append(entry["pca"])
        colors.append(top_labels.index(label))  # assign integer color
        labels_used.append(label)

umap_coords = np.array(umap_coords)
pca_coords = np.array(pca_coords)
colors = np.array(colors)

# Define color map
cmap = get_cmap("tab10")

# Plot UMAP
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(umap_coords[:, 0], umap_coords[:, 1], c=colors, cmap=cmap, s=10, alpha=0.6)
plt.title("UMAP Colored by Label (Top 10)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.grid(True)

# Plot PCA
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(pca_coords[:, 0], pca_coords[:, 1], c=colors, cmap=cmap, s=10, alpha=0.6)
plt.title("PCA Colored by Label (Top 10)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)

# Add legend
handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                      markerfacecolor=cmap(i), markersize=6)
           for i, label in enumerate(top_labels)]
plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', title="Labels")

plt.tight_layout()
plt.show()
