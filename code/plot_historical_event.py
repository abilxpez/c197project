import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from datetime import datetime

# Load the data
file_path = "chat_label_checkpoint_congress.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Historical events
us_history_events = {
    1996: ["Welfare Reform Act signed", "Telecommunications Act passed", "Clinton re-elected"],
    1998: ["Clinton impeachment begins", "Google founded"],
    1999: ["Columbine shooting"],
    2000: ["Bush v. Gore decision", "Dot-com bubble bursts"],
    2001: ["9/11 attacks", "War in Afghanistan begins", "PATRIOT Act signed"],
    2003: ["Iraq War begins", "DHS established"],
    2005: ["Hurricane Katrina"],
    2007: ["Great Recession begins"],
    2008: ["Obama elected", "Financial crisis"],
    2009: ["Stimulus Act passed", "ACA work begins"],
    2010: ["ACA signed", "BP oil spill"],
    2011: ["Bin Laden killed", "Occupy Wall Street"],
    2012: ["Obama re-elected", "Sandy Hook shooting"],
    2013: ["Snowden leaks", "Boston bombing"],
    2014: ["ISIS airstrikes", "Ferguson protests"]
}

# Count labels and speeches per year
label_year_counts = defaultdict(lambda: defaultdict(int))
total_speeches_per_year = defaultdict(int)

for speech in data:
    try:
        year = datetime.strptime(speech["date"], "%Y-%m-%d").year
        total_speeches_per_year[year] += 1
        for label in speech.get("labels", []):
            label_year_counts[label][year] += 1
    except (KeyError, ValueError):
        continue

# Calculate proportions
label_year_proportions = defaultdict(lambda: defaultdict(float))
for label, year_counts in label_year_counts.items():
    for year, count in year_counts.items():
        total = total_speeches_per_year.get(year, 1)
        label_year_proportions[label][year] = count / total

# Convert to DataFrames
df_counts = pd.DataFrame(label_year_counts).fillna(0).astype(int)
df_proportions = pd.DataFrame(label_year_proportions).fillna(0).sort_index()

# Top 10 labels by total count
top_labels = df_counts.sum().sort_values(ascending=False).head(10).index
df_prop_top = df_proportions[top_labels]

# Plotting
fig, ax = plt.subplots(figsize=(16, 9))
df_prop_top.plot(ax=ax, kind='line', marker='o')

# Add historical event annotations
for year, events in us_history_events.items():
    if year in df_prop_top.index:
        ax.axvline(x=year, color='gray', linestyle='--', alpha=0.5)
        y_base = df_prop_top.max().max() * 0.95  # Start near the top
        for i, event in enumerate(events):
            ax.annotate(
                event,
                xy=(year, y_base - i * 0.05),
                xytext=(year + 0.3, y_base - i * 0.05),
                textcoords='data',
                fontsize=8,
                va='top',
                arrowprops=dict(arrowstyle='-', lw=0.5, color='gray')
            )

# Final touches
ax.set_title("Proportion of Speeches by Label per Year with Historical Events (Top 10 Labels)")
ax.set_xlabel("Year")
ax.set_ylabel("Proportion of Speeches")
ax.legend(title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
