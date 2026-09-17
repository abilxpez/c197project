import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from datetime import datetime

# Load the data
file_path = "merged_labeled_speeches.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Select potentially partisan topics
selected_labels = [
    "Economy and Trade",
    "National Defense and Military",
    "Foreign Policy and Diplomacy",
    "Immigration and Border Policy",
]

# Count label frequencies by year and party
party_label_year_counts = {
    "D": defaultdict(lambda: defaultdict(int)),
    "R": defaultdict(lambda: defaultdict(int)),
}
party_total_year_counts = {
    "D": defaultdict(int),
    "R": defaultdict(int),
}

# Process speeches
for speech in data:
    try:
        year = datetime.strptime(speech["date"], "%Y-%m-%d").year
        party = speech.get("speaker_party")
        if party not in ["D", "R"]:
            continue
        party_total_year_counts[party][year] += 1
        for label in speech.get("labels", []):
            if label in selected_labels:
                party_label_year_counts[party][label][year] += 1
    except (KeyError, ValueError):
        continue

# Convert to DataFrames (proportions)
dfs_party = {}
for party in ["D", "R"]:
    prop_dict = defaultdict(dict)
    for label in selected_labels:
        for year in party_label_year_counts[party][label]:
            total = party_total_year_counts[party][year]
            count = party_label_year_counts[party][label][year]
            prop_dict[label][year] = count / total if total > 0 else 0
    dfs_party[party] = pd.DataFrame(prop_dict).sort_index().fillna(0)

# Plot results
fig, axes = plt.subplots(len(selected_labels), 1, figsize=(14, 3 * len(selected_labels)), sharex=True)

for i, label in enumerate(selected_labels):
    axes[i].plot(dfs_party["D"].index, dfs_party["D"][label], label="Democrats", marker='o', color='blue')
    axes[i].plot(dfs_party["R"].index, dfs_party["R"][label], label="Republicans", marker='s', color='red')
    axes[i].set_title(f"Proportion of Speeches about '{label}'")
    axes[i].set_ylabel("Proportion")
    axes[i].grid(True)
    axes[i].legend()

# Force every year to be shown on x-axis
axes[-1].set_xlabel("Year")
axes[-1].set_xticks(dfs_party["D"].index)
plt.tight_layout()
plt.show()
