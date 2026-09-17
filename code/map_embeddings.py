import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load the combined .npz file
data = np.load("congress_speech_embeddings_100k_combined.npz", allow_pickle=True)
dates = data["dates"]

# Convert string dates (e.g., "1996-03-13") to datetime objects
parsed_dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates]

# Plot a histogram by year
years = [d.year for d in parsed_dates]

plt.figure(figsize=(12, 6))
plt.hist(years, bins=range(min(years), max(years) + 1), edgecolor='black')
plt.title("Distribution of Congressional Speech Embedding Dates")
plt.xlabel("Year")
plt.ylabel("Number of Speeches")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
