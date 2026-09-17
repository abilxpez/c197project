import json
from datetime import datetime

# Load data
file_path = "merged_labeled_speeches.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract all valid dates
dates = []
for entry in data:
    try:
        dt = datetime.strptime(entry["date"], "%Y-%m-%d")
        dates.append(dt)
    except Exception:
        continue

# Find earliest and latest
if dates:
    earliest = min(dates)
    latest = max(dates)
    print(f"Earliest speech date: {earliest.date()}")
    print(f"Latest speech date: {latest.date()}")
else:
    print("No valid dates found.")
