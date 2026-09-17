import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Load data
with open("merged_labeled_speeches_umap_pca.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract UMAP, PCA, and year
umap_coords = []
pca_coords = []
years = []

for entry in data:
    try:
        year = int(entry["date"][:4])
        umap_coords.append(entry["umap"])
        pca_coords.append(entry["pca"])
        years.append(year)
    except:
        continue  # Skip entries with missing or malformed data

umap_coords = np.array(umap_coords)
pca_coords = np.array(pca_coords)
years = np.array(years)

# Normalize years for color mapping
norm = plt.Normalize(years.min(), years.max())
cmap = cm.get_cmap("viridis")

# Plot UMAP
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(umap_coords[:, 0], umap_coords[:, 1], c=cmap(norm(years)), s=10, alpha=0.7)
plt.title("UMAP Projection Colored by Year")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.grid(True)

# Plot PCA
plt.subplot(1, 2, 2)
plt.scatter(pca_coords[:, 0], pca_coords[:, 1], c=cmap(norm(years)), s=10, alpha=0.7)
plt.title("PCA Projection Colored by Year")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)

# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=plt.gcf().get_axes(), label="Year", orientation="vertical", fraction=0.025)

plt.tight_layout()
plt.show()
