import json
from collections import Counter
import matplotlib.pyplot as plt

# Load the data
file_path = "merged_labeled_speeches.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Count speeches per state
state_counter = Counter()
for entry in data:
    state = entry.get("speaker_state", "Unknown")
    state_counter[state] += 1

# Sort states by number of speeches
sorted_states = state_counter.most_common()
states, counts = zip(*sorted_states)

# Plotting
plt.figure(figsize=(14, 6))
plt.bar(states, counts)
plt.xticks(rotation=75, ha='right')
plt.ylabel("Number of Speeches")
plt.title("Number of Speeches per State")
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
