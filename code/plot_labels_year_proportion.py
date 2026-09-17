import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from datetime import datetime

# Load the data from the file
file_path = "merged_labeled_speeches.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Accumulate label counts per year
label_year_counts = defaultdict(lambda: defaultdict(int))
total_speeches_per_year = defaultdict(int)

for speech in data:
    try:
        year = datetime.strptime(speech["date"], "%Y-%m-%d").year
        total_speeches_per_year[year] += 1
        for label in speech.get("labels", []):
            label_year_counts[label][year] += 1
    except (KeyError, ValueError):
        continue  # Skip if date format is wrong or labels are missing

# Compute proportions per label per year
label_year_proportions = defaultdict(lambda: defaultdict(float))
for label, year_counts in label_year_counts.items():
    for year, count in year_counts.items():
        total = total_speeches_per_year.get(year, 1)  # Avoid division by zero
        label_year_proportions[label][year] = count / total

# Convert to DataFrame
df_counts = pd.DataFrame(label_year_counts).fillna(0).astype(int)
df_proportions = pd.DataFrame(label_year_proportions).fillna(0)

# Ensure index is sorted and cast to integers
df_proportions.index = df_proportions.index.astype(int)
df_proportions = df_proportions.sort_index()

# === Label Selection ===

# Manually select labels to analyze
selected_labels = [
    "Economy and Trade",
    "Labor, Jobs, and Workers' Rights",
    "Healthcare and Public Health",
    "Terrorism, Homeland Security, and War on Terror",
    "Civil Rights and Racial Equality"
]

# labels_to_plot = selected_labels

# OR: Automatically plot bottom N labels by total count
top_n = 12
labels_to_plot = df_counts.sum().sort_values(ascending=False).index


# Split into top 10, middle 10, and bottom 2
top_labels = labels_to_plot[:8]
middle_labels = labels_to_plot[8:15]
bottom_labels = labels_to_plot[15:]

# Filter DataFrame
df_prop_selected = df_proportions[bottom_labels]

# Plotting
fig, ax = plt.subplots(figsize=(14, 8))
df_prop_selected.plot(ax=ax, kind='line', marker='o')
ax.set_title("Proportion of Speeches by Label per Year - Bottom 7 Labels")
ax.set_xlabel("Year")
ax.set_ylabel("Proportion of Speeches")
ax.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(ticks=df_prop_selected.index, labels=df_prop_selected.index, rotation=45)
plt.tight_layout()
plt.grid(True)

plt.show()
