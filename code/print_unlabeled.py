import json
import random

# File paths
labeled_path = "newly_labeled_speeches.json"
recovered_path = "congress_speeches_recovered.json"
unlabeled_ids_path = "unlabeled_ids.json"

# Load labeled data and collect only the IDs with "Unlabeled"
with open(labeled_path, "r", encoding="utf-8") as f:
    labeled_data = json.load(f)
    unlabeled_ids = [entry["id"] for entry in labeled_data if "Unlabeled" in entry.get("labels", [])]

# Save unlabeled IDs to JSON
with open(unlabeled_ids_path, "w", encoding="utf-8") as f:
    json.dump(unlabeled_ids, f, indent=2)
print(f"Saved {len(unlabeled_ids)} unlabeled IDs to {unlabeled_ids_path}")

# Randomly sample 10 IDs
sample_ids = random.sample(unlabeled_ids, min(10, len(unlabeled_ids)))

# Load recovered data into a dictionary for quick lookup
with open(recovered_path, "r", encoding="utf-8") as f:
    recovered_data = {entry["id"]: entry for entry in json.load(f)}

# Print sampled speeches
for i, doc_id in enumerate(sample_ids, 1):
    speech = recovered_data.get(doc_id)
    if speech:
        print(f"\n=== Speech {i} ===")
        print(f"ID: {speech['id']}")
        print(f"Title: {speech.get('title', '(No Title)')}")
        print(f"Text:\n{speech['text'][:1000]}{'...' if len(speech['text']) > 1000 else ''}")
