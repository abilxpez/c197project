import json
import numpy as np

# Load the labeled speeches
with open("congress_sampled_labeled_speeches.json", "r") as f:
    labeled_data = json.load(f)

# Collect IDs of labeled speeches
labeled_ids = set(speech["id"] for speech in labeled_data)

# Load the 100k embeddings
data = np.load("congress_speech_embeddings_100k_combined.npz", allow_pickle=True)
doc_ids = data["doc_ids"]
dates = data["dates"]
embeddings = data["embeddings"]

# Filter out any entries whose ID is in labeled_ids
filtered_doc_ids = []
filtered_dates = []
filtered_embeddings = []

for i in range(len(doc_ids)):
    if doc_ids[i] not in labeled_ids:
        filtered_doc_ids.append(doc_ids[i])
        filtered_dates.append(dates[i])
        filtered_embeddings.append(embeddings[i])

# Save filtered results to a new file
filtered_file = "congress_embeddings_99k_filtered.npz"
np.savez(
    filtered_file,
    doc_ids=np.array(filtered_doc_ids),
    dates=np.array(filtered_dates),
    embeddings=np.array(filtered_embeddings)
)

print(f"Original: {len(doc_ids)}")
print(f"Removed: {len(doc_ids) - len(filtered_doc_ids)}")
print(f"Remaining: {len(filtered_doc_ids)}")
