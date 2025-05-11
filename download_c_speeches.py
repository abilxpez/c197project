import os
import requests
import json
import numpy as np
from time import time, sleep
from datetime import timedelta

BASE_URL = "http://congressionalspeech.lib.uiowa.edu/api.php/speeches"
OUTPUT_JSON = "congress_speeches.json"
OUTPUT_NPZ = "congress_speeches.npz"
CHECKPOINT_FILE = "congress_checkpoint.json"

PAGE_SIZE = 50
START_PAGE = 1

# Load checkpoint if it exists
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        checkpoint = json.load(f)
        all_data = checkpoint["data"]
        current_page = checkpoint["last_page"] + 1
        print(f"Resuming from page {current_page}...")
else:
    all_data = []
    current_page = START_PAGE

start_time = time()

while True:
    print(f"\nFetching page {current_page}...")
    url = f"{BASE_URL}?transform=1&order=id&page={current_page},{PAGE_SIZE}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Request failed: {response.status_code}")
        break

    page_data = response.json()
    if isinstance(page_data, dict) and "speeches" in page_data:
        page_data = page_data["speeches"]
    elif not isinstance(page_data, list):
        print("Unexpected response format.")
        break

    if not page_data:
        print("No more data returned.")
        break

    # Filter fields and preview speech
    for entry in page_data:
        filtered_entry = {
            "id": entry["id"],
            "date": entry.get("date", ""),
            "title": entry.get("title", ""),
            "text": entry.get("speaking", ""),
            "speaker_state": entry.get("speaker_state", ""),
            "speaker_party": entry.get("speaker_party", "")
        }

        preview = ' '.join(filtered_entry["text"].split()[:50])
        print(f"[{filtered_entry['id']}] {filtered_entry['title'][:50]}... — {preview}...\n")

        all_data.append(filtered_entry)

    # Save checkpoint
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_page": current_page, "data": all_data}, f)

    # Estimate remaining time
    elapsed = time() - start_time
    pages_done = current_page - START_PAGE + 1
    avg_time = elapsed / pages_done
    est_remaining = timedelta(seconds=int(avg_time * 1000 - elapsed))  # guess 1000 pages
    print(f"Page {current_page} fetched. Estimated remaining: {est_remaining}")

    current_page += 1
    sleep(0.5)

# Save to JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(all_data, f, indent=2)

# Save to .npz
np.savez(
    OUTPUT_NPZ,
    ids=np.array([d["id"] for d in all_data]),
    dates=np.array([d["date"] for d in all_data]),
    titles=np.array([d["title"] for d in all_data]),
    texts=np.array([d["text"] for d in all_data]),
    speaker_states=np.array([d["speaker_state"] for d in all_data]),
    speaker_parties=np.array([d["speaker_party"] for d in all_data]),
)

# Cleanup checkpoint
try:
    os.remove(CHECKPOINT_FILE)
except Exception as e:
    print(f"Could not delete checkpoint: {e}")

print(f"\nSaved speeches to '{OUTPUT_JSON}' and '{OUTPUT_NPZ}'")
