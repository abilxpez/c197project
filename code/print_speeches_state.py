import json
from collections import Counter

# Load the data
with open("chat_label_checkpoint_congress.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Track state counts and missing values
state_counts = Counter()
missing_state_entries = 0

for entry in data:
    state = entry.get("speaker_state")
    if state and isinstance(state, str) and state.strip() != "":
        state_counts[state.strip()] += 1
    else:
        missing_state_entries += 1

# Full list of U.S. state abbreviations (including DC)
all_states = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY", "DC"
}

# Print speeches per state
print("Speeches per state:")
for state, count in sorted(state_counts.items()):
    print(f"{state}: {count}")

# Print total
print(f"\nTotal speeches: {len(data)}")
print(f"Total speeches with missing or invalid state: {missing_state_entries}")

# Print states with 0 speeches
missing_states = sorted(all_states - set(state_counts.keys()))
print("\nStates with 0 speeches:")
for state in missing_states:
    print(state)