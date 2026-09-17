import json
from collections import Counter
import matplotlib.pyplot as plt

def plot_label_distribution(label_counter):
    # Remove "Unlabeled" before plotting
    filtered = {label: count for label, count in label_counter.items() if label != "Unlabeled"}
    labels, counts = zip(*sorted(filtered.items(), key=lambda x: x[1], reverse=True))
    
    plt.figure(figsize=(12, 6))
    plt.bar(labels, counts)
    plt.xticks(rotation=75, ha='right')
    plt.ylabel("Number of Speeches")
    plt.title("Model Labeled - Number of Speeches per Label (Excluding 'Unlabeled')")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()

# Load predicted data
with open("congress_speeches_recovered.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Count labels
label_counter = Counter()
for entry in data:
    labels = entry.get("labels", [])
    label_counter.update(labels)

# Print total number of speeches
print(f"\nTotal number of speeches: {len(data)}\n")

# Print results sorted by count
print("Number of speeches per label:\n")
for label, count in label_counter.most_common():
    print(f"{label}: {count}")

# Plot graph (excluding "Unlabeled")
plot_label_distribution(label_counter)
