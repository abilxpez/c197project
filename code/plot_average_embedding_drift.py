import json
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

# Load the data
with open("merged_labeled_speeches.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract embeddings and years
year_embeddings = defaultdict(list)
all_embeddings = []

for entry in data:
    embedding = entry.get("embedding")
    if embedding:
        try:
            year = datetime.strptime(entry["date"], "%Y-%m-%d").year
            embedding_np = np.array(embedding)
            year_embeddings[year].append(embedding_np)
            all_embeddings.append(embedding_np)
        except Exception:
            continue  # Skip malformed entries

# Compute the global centroid
global_centroid = np.mean(all_embeddings, axis=0)

# Compute cosine distance of each year's centroid to the global centroid
years = sorted(year_embeddings)
semantic_drift = {}

for year in years:
    yearly_embeddings = np.vstack(year_embeddings[year])
    year_centroid = np.mean(yearly_embeddings, axis=0).reshape(1, -1)
    distance = cosine_distances(year_centroid, global_centroid.reshape(1, -1))[0][0]
    semantic_drift[year] = distance

# Plot the semantic drift over time
plt.figure(figsize=(12, 6))
plt.plot(list(semantic_drift.keys()), list(semantic_drift.values()), marker='o')
plt.title("Average Embedding Drift Over Time")
plt.xlabel("Year")
plt.ylabel("Cosine Distance from Global Centroid")
plt.grid(True)
plt.tight_layout()
plt.show()
