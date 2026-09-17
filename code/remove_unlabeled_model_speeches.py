import json

# File paths
input_file = "congress_sampled_labeled_speeches_embedded_10k.json"
output_file = "chat_labeled_embedded_speeches_10k.json"

# Load data
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Filter out speeches with label "Unlabeled"
filtered_data = [
    entry for entry in data
    if "Unlabeled" not in entry.get("labels", [])
]

# Save to new file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent=2)

print(f"Saved {len(filtered_data)} speeches to {output_file}")
