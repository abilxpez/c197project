import json

# File paths
recovered_path = "congress_speeches_recovered.json"
partial_path = "model_labeled_speeches_no_state_party.json"
output_path = "model_labeled_speeches.json"

# Load recovered speeches with state and party info
with open(recovered_path, "r", encoding="utf-8") as f:
    recovered = {entry["id"]: entry for entry in json.load(f)}

# Load labeled speeches without state/party
with open(partial_path, "r", encoding="utf-8") as f:
    labeled = json.load(f)

# Merge speaker_state and speaker_party based on id
merged = []
for entry in labeled:
    speech_id = entry["id"]
    if speech_id in recovered:
        entry["speaker_state"] = recovered[speech_id].get("speaker_state")
        entry["speaker_party"] = recovered[speech_id].get("speaker_party")
        merged.append(entry)
    else:
        print(f"Warning: ID {speech_id} not found in recovered data")

# Save merged data
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)

print(f"Saved {len(merged)} entries to {output_path}")
