import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

# Load the NPZ file
data = np.load("congress_embeddings_99k_filtered.npz", allow_pickle=True)
dates = data["dates"]

# Convert string dates to datetime objects (assumes format YYYY-MM-DD)
parsed_dates = []
for d in dates:
    try:
        parsed_dates.append(datetime.strptime(d, "%Y-%m-%d"))
    except:
        continue  # Skip malformed dates

# Extract just the year from each date
years = [d.year for d in parsed_dates]

# Count how many speeches per year
year_counts = Counter(years)
sorted_years = sorted(year_counts.items())
x, y = zip(*sorted_years)

# Plot
plt.figure(figsize=(12, 6))
plt.bar(x, y)
plt.xlabel("Year")
plt.ylabel("Number of Speeches")
plt.title("Distribution of Speeches Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
