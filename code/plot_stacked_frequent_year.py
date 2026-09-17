import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np

# Load data
file_path = "merged_labeled_speeches.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Count label frequencies per year
label_counts_by_year = defaultdict(Counter)
for entry in data:
    year = entry["date"][:4]
    for label in entry.get("labels", []):
        label_counts_by_year[year][label] += 1

# Get sorted list of years and most common labels
all_years = sorted(label_counts_by_year.keys())
all_labels = Counter()
for yearly_counter in label_counts_by_year.values():
    all_labels.update(yearly_counter)

# Get top labels in most-to-least frequent order
top_labels = [label for label, _ in all_labels.most_common(10)]

# Build matrix: rows = labels, columns = years
label_year_matrix = np.zeros((len(top_labels), len(all_years)))
for i, label in enumerate(top_labels):
    for j, year in enumerate(all_years):
        label_year_matrix[i, j] = label_counts_by_year[year][label]

# Reverse only the matrix to place most frequent label on top
reversed_matrix = label_year_matrix

# Plot stacked area chart
plt.figure(figsize=(14, 6))
plt.stackplot(all_years, reversed_matrix, labels=top_labels)
plt.title("Top 10 Labels Over Time (Stacked Area Plot)")
plt.xlabel("Year")
plt.ylabel("Number of Speeches")
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), title="Labels")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.show()
