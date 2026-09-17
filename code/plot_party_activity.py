import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

# Load the data
with open("merged_labeled_speeches.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Normalize party labels
def normalize_party(party):
    if party in ["D", "Democrat"]:
        return "Democrat"
    elif party in ["R", "Republican"]:
        return "Republican"
    return "Other"

# Count speeches per year per normalized party
party_year_counts = defaultdict(Counter)
for entry in data:
    year = datetime.strptime(entry["date"], "%Y-%m-%d").year
    party = normalize_party(entry.get("speaker_party", "Unknown"))
    party_year_counts[party][year] += 1

# Get all years across parties for complete x-axis
all_years = sorted({year for counts in party_year_counts.values() for year in counts})

# Plot
plt.figure(figsize=(12, 6))
for party in ["Democrat", "Republican"]:
    counts = party_year_counts.get(party, {})
    frequencies = [counts.get(year, 0) for year in all_years]
    plt.plot(all_years, frequencies, label=party)

plt.title("Number of Speeches per Year by Party")
plt.xlabel("Year")
plt.ylabel("Number of Speeches")
plt.xticks(all_years, rotation=45)  # Force every year to show
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
