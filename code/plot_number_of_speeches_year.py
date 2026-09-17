import json
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

# Load the merged JSON data
file_path = "merged_labeled_speeches_umap_pca.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Count speeches per year
year_counter = Counter()
for entry in data:
    try:
        year = datetime.strptime(entry["date"], "%Y-%m-%d").year
        year_counter[year] += 1
    except Exception:
        continue  # Skip invalid or missing dates

# Sort years and corresponding counts
sorted_years = sorted(year_counter)
counts = [year_counter[year] for year in sorted_years]

# Print the count for each year
print("Number of speeches per year:")
for year in sorted_years:
    print(f"{year}: {year_counter[year]}")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(sorted_years, counts, marker='o')
plt.title("Number of Speeches per Year")
plt.xlabel("Year")
plt.ylabel("Number of Speeches")
plt.xticks(sorted_years, rotation=45)  # Show every year on x-axis
plt.grid(True)
plt.tight_layout()
plt.show()
