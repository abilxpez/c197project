import json

# File paths
chat_file = "chat_labeled_embedded_speeches_8k.json"
model_file = "model_labeled_speeches_44k.json"
output_file = "merged_labeled_speeches.json"

# Load both datasets
with open(chat_file, "r", encoding="utf-8") as f:
    chat_data = json.load(f)

with open(model_file, "r", encoding="utf-8") as f:
    model_data = json.load(f)

# Prepare merged list and set for tracking duplicates
merged = []
seen_ids = set()

# Helper function to keep only the desired fields
def clean_entry(entry):
    return {
        "id": entry["id"],
        "date": entry["date"],
        "embedding": entry["embedding"],
        "labels": entry["labels"],
        "speaker_state": entry["speaker_state"],
        "speaker_party": entry["speaker_party"]
    }

# Add entries from chat_data
for entry in chat_data:
    if entry["id"] in seen_ids:
        print(f"Duplicate ID found: {entry['id']}")
    else:
        merged.append(clean_entry(entry))
        seen_ids.add(entry["id"])

# Add entries from model_data
for entry in model_data:
    if entry["id"] in seen_ids:
        print(f"Duplicate ID found: {entry['id']}")
    else:
        merged.append(clean_entry(entry))
        seen_ids.add(entry["id"])

# Save merged output
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)

print(f"\nMerged dataset saved to: {output_file}")
