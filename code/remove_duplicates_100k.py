import json
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "keep"
LABELED_FILE = DATA_DIR / "congress_sampled_labeled_speeches_embedded_10k.json"
FULL_EMBEDDINGS_FILE = DATA_DIR / "congress_speech_embeddings_100k_combined.npz"
FILTERED_EMBEDDINGS_FILE = DATA_DIR / "congress_speech_embeddings_99k_filtered.npz"

# Load the ChatGPT-labeled training speeches.
with open(LABELED_FILE, "r") as f:
    labeled_data = json.load(f)

# Collect IDs that should not be included in the unlabeled prediction pool.
labeled_ids = set(speech["id"] for speech in labeled_data)

# Load the full combined 100k embeddings.
data = np.load(FULL_EMBEDDINGS_FILE, allow_pickle=True)
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

np.savez(
    FILTERED_EMBEDDINGS_FILE,
    doc_ids=np.array(filtered_doc_ids),
    dates=np.array(filtered_dates),
    embeddings=np.array(filtered_embeddings)
)

print(f"Original: {len(doc_ids)}")
print(f"Removed: {len(doc_ids) - len(filtered_doc_ids)}")
print(f"Remaining: {len(filtered_doc_ids)}")
print(f"Saved: {FILTERED_EMBEDDINGS_FILE}")
