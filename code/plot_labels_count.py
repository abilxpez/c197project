import json
from collections import Counter
import matplotlib.pyplot as plt

# File path to your merged data
file_path = "merged_labeled_speeches.json"

# Load the data
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Count labels
label_counter = Counter()
for entry in data:
    labels = entry.get("labels", [])
    label_counter.update(labels)

# Sort labels by count (descending)
sorted_labels = label_counter.most_common()

# Split into labels and counts
labels, counts = zip(*sorted_labels)

# Plotting
plt.figure(figsize=(14, 6))
plt.bar(labels, counts)
plt.xticks(rotation=75, ha='right')
plt.ylabel("Number of Speeches")
plt.title("Label Counts")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
