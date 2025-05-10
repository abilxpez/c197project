import requests
import json
import numpy as np

BASE_URL = "http://congressionalspeech.lib.uiowa.edu/api.php/speeches"
OUTPUT_JSON = "test_congress_speeches.json"
OUTPUT_NPZ = "test_congress_speeches.npz"

PAGE_SIZE = 5
page = 1

# Fetch 1 page of 5 speeches
print(f"Fetching {PAGE_SIZE} speeches from page {page}...")
url = f"{BASE_URL}?transform=1&order=id&page={page},{PAGE_SIZE}"
response = requests.get(url)

if response.status_code != 200:
    raise Exception(f"Request failed: {response.status_code}")

data = response.json()

# Defensive check for format
if isinstance(data, dict) and "speeches" in data:
    data = data["speeches"]
elif not isinstance(data, list):
    print("Unexpected response format. Here's what we got:")
    print(json.dumps(data, indent=2))
    raise Exception("Data is not a list or valid 'speeches' object.")

print(f"Retrieved {len(data)} speeches.")

# Keep only relevant fields
filtered_data = []
for entry in data:
    filtered_entry = {
        "id": entry["id"],
        "date": entry.get("date", ""),
        "title": entry.get("title", ""),
        "text": entry.get("speaking", ""),
        "speaker_state": entry.get("speaker_state", ""),
        "speaker_party": entry.get("speaker_party", "")
    }
    filtered_data.append(filtered_entry)

# Show preview of first cleaned entry
print("\nFirst speech (filtered):")
print(json.dumps(filtered_data[0], indent=2))

# Save to JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(filtered_data, f, indent=2)

# Save to .npz
np.savez(
    OUTPUT_NPZ,
    ids=np.array([d["id"] for d in filtered_data]),
    dates=np.array([d["date"] for d in filtered_data]),
    titles=np.array([d["title"] for d in filtered_data]),
    texts=np.array([d["text"] for d in filtered_data]),
    speaker_states=np.array([d["speaker_state"] for d in filtered_data]),
    speaker_parties=np.array([d["speaker_party"] for d in filtered_data]),
)

print(f"\nSaved test speeches to '{OUTPUT_JSON}' and '{OUTPUT_NPZ}'")
