# Re-import after execution state reset
import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

# Find the most frequent label per year
top_label_per_year = {}
for year, counter in label_counts_by_year.items():
    if counter:
        top_label, count = counter.most_common(1)[0]
        top_label_per_year[int(year)] = (top_label, count)

# Sort by year
sorted_years = sorted(top_label_per_year.keys())
top_labels = [top_label_per_year[year][0] for year in sorted_years]
counts = [top_label_per_year[year][1] for year in sorted_years]

# Assign a unique color to each label
unique_labels = list(set(top_labels))
label_to_color = {label: plt.cm.tab20(i % 20) for i, label in enumerate(unique_labels)}
bar_colors = [label_to_color[label] for label in top_labels]

# Plot
plt.figure(figsize=(14, 6))
bars = plt.bar(sorted_years, counts, color=bar_colors)
plt.title("Most Frequent Label per Year")
plt.xlabel("Year")
plt.ylabel("Number of Speeches")
plt.xticks(rotation=45)

# Add legend
legend_handles = [mpatches.Patch(color=color, label=label) for label, color in label_to_color.items()]
plt.legend(handles=legend_handles, title="Top Labels", bbox_to_anchor=(1.05, 1), loc='upper left')

# Apply layout after adding the legend
plt.tight_layout()
plt.show()
