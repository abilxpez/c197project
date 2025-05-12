import numpy as np
from collections import defaultdict
from sklearn.preprocessing import normalize

# Step 1: Load embeddings
print("Loading embeddings...")
embeddings_data = np.load("president_speech_embeddings.npz", allow_pickle=True)
doc_names = embeddings_data["doc_names"]
embeddings = embeddings_data["embeddings"]
print(f"Loaded {len(doc_names)} embeddings.")

# Step 2: Load labels
print("Loading labels...")
labels_data = np.load("president_speech_labels.npz", allow_pickle=True)
label_doc_names = labels_data["doc_names"]
label_labels = labels_data["labels"]
print(f"Loaded {len(label_doc_names)} labeled documents.")

# Step 3: Build mapping from doc_name to labels
doc_to_labels = {name: lbl for name, lbl in zip(label_doc_names, label_labels)}

# Step 4: Group embeddings by label
print("Grouping embeddings by label...")
label_to_embeddings = defaultdict(list)
missing_count = 0

for doc_name, emb in zip(doc_names, embeddings):
    labels = doc_to_labels.get(doc_name)
    if labels is None:
        missing_count += 1
        continue
    for label in labels:
        label_to_embeddings[label].append(emb)

print(f"Grouped embeddings for {len(label_to_embeddings)} labels.")
if missing_count > 0:
    print(f"Skipped {missing_count} documents with no labels.")

# Step 5: Compute normalized centroids
print("Computing normalized label centroids...")
label_centroids = {
    label: normalize(np.mean(vecs, axis=0, keepdims=True))[0]
    for label, vecs in label_to_embeddings.items()
}
print("Finished computing centroids.")

# Step 6: Save to .npz file
output_path = "label_centroids_normalized.npz"
np.savez(output_path, **label_centroids)
print(f"Saved normalized centroids to '{output_path}'.")
