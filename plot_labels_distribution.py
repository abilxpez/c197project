import json
from collections import Counter
import matplotlib.pyplot as plt

# Load your local JSON file
with open("sampled_labeled_speeches.json", "r") as f:
    speeches = json.load(f)

# Count how often each label appears
label_counter = Counter()
for speech in speeches:
    labels = speech.get("labels", [])
    label_counter.update(labels)

# Sort labels by frequency
sorted_labels = sorted(label_counter.items(), key=lambda x: x[1], reverse=True)
labels, counts = zip(*sorted_labels)

# Plot the label distribution
plt.figure(figsize=(12, 6))
plt.bar(labels, counts)
plt.xticks(rotation=75, ha='right')
plt.title("Label Distribution in Sampled Congressional Speeches")
plt.ylabel("Number of Speeches")
plt.tight_layout()
plt.show()
