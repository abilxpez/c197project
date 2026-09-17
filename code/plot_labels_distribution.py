import json
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from datetime import datetime

# Load the JSON data
with open("merged_labeled_speeches.json", "r") as f:
    speeches = json.load(f)

# Create a mapping: label -> Counter(year)
label_year_counts = defaultdict(Counter)
label_total_counts = Counter()

for speech in speeches:
    date_str = speech.get("date", "")
    try:
        year = int(date_str[:4])
    except:
        continue  # skip invalid/missing dates

    for label in speech.get("labels", []):
        label_year_counts[label][year] += 1
        label_total_counts[label] += 1

# Topic labels (for reference)
topic_labels = [
    "Economy and Trade",
    "National Defense and Military",
    "Foreign Policy and Diplomacy",
    "Immigration and Border Policy",
    "Civil Rights and Racial Equality",
    "Women's Rights",
    "LGBTQ+ Rights",
    "Law Enforcement and Criminal Justice",
    "Healthcare and Public Health",
    "Education and Schools",
    "Science, Technology, and Innovation",
    "Climate and Environment",
    "Infrastructure and Transportation",
    "Government Reform and Corruption",
    "Elections and Democratic Institutions",
    "Religion, Values, and National Identity",
    "Social Welfare and Poverty",
    "Labor, Jobs, and Workers' Rights",
    "Gun Policy and Second Amendment",
    "Energy and Natural Resources",
    "Terrorism, Homeland Security, and War on Terror",
    "Indigenous and Tribal Affairs"
]

# Select potentially partisan topics
selected_labels = [
    "Economy and Trade",
    "Labor, Jobs, and Workers' Rights",
    "Healthcare and Public Health",
    "Terrorism, Homeland Security, and War on Terror",
    "Civil Rights and Racial Equality"
]

# ====== You can switch the label selection below ======
# Use this to manually pick labels:
labels_to_plot = selected_labels

# Or comment the above and use this for top 10:
# sorted_labels = label_total_counts.most_common()
# labels_to_plot = [label for label, _ in sorted_labels[:10]]

# Or use this for bottom 12:
# labels_to_plot = [label for label, _ in sorted_labels[-12:]]
# ======================================================

# Plot selected labels
plt.figure(figsize=(14, 7))
for label in labels_to_plot:
    year_count = label_year_counts[label]
    years = sorted(year_count)
    counts = [year_count[year] for year in years]
    plt.plot(years, counts, label=label)

plt.title("Speech Label Distribution Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Speeches")
plt.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.grid(True)
plt.show()
