import os
import json
import requests
import numpy as np
from time import sleep

BASE_URL = "http://congressionalspeech.lib.uiowa.edu/api.php/speeches"
CHECKPOINT_FILE = "download_checkpoint.json"
CHUNK_SIZE = 50000
PAGE_SIZE = 50

# Initialize
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        checkpoint = json.load(f)
    current_page = checkpoint["last_page"] + 1
    total_downloaded = checkpoint["total_downloaded"]
    chunk_index = checkpoint["chunk_index"]
    print(f"Resuming from page {current_page}...")
else:
    current_page = 4693  # 234646 / 50 = ~4692.92
    total_downloaded = 234646
    chunk_index = 0
    print("Starting from page", current_page)

buffer = []

def save_chunk(data, index):
    suffix = str(index).zfill(5)
    json_file = f"congress_chunk_{suffix}.json"
    npz_file = f"congress_chunk_{suffix}.npz"

    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)

    np.savez(
        npz_file,
        ids=np.array([d["id"] for d in data]),
        dates=np.array([d["date"] for d in data]),
        titles=np.array([d["title"] for d in data]),
        texts=np.array([d["text"] for d in data]),
        speaker_states=np.array([d["speaker_state"] for d in data]),
        speaker_parties=np.array([d["speaker_party"] for d in data]),
    )

    print(f"Saved chunk {index} to {json_file} and {npz_file}")

# Download loop
while True:
    print(f"\nFetching page {current_page}...")
    url = f"{BASE_URL}?transform=1&order=id&page={current_page},{PAGE_SIZE}"
    resp = requests.get(url)

    if resp.status_code != 200:
        print(f"Failed to fetch page {current_page}: {resp.status_code}")
        break

    page_data = resp.json()
    if isinstance(page_data, dict) and "speeches" in page_data:
        page_data = page_data["speeches"]
    elif not isinstance(page_data, list):
        print("Unexpected response format.")
        break

    if not page_data:
        print("No more data.")
        break

    for entry in page_data:
        speech = {
            "id": entry["id"],
            "date": entry.get("date", ""),
            "title": entry.get("title", ""),
            "text": entry.get("speaking", ""),
            "speaker_state": entry.get("speaker_state", ""),
            "speaker_party": entry.get("speaker_party", "")
        }
        buffer.append(speech)
        total_downloaded += 1

        if total_downloaded % 50 == 0:
            preview = ' '.join(speech["text"].split()[:20])
            print(f"[{speech['id']}] {speech['title'][:40]} — {preview}...")

        if len(buffer) == CHUNK_SIZE:
            save_chunk(buffer, chunk_index)
            buffer = []
            chunk_index += 1

    # Save checkpoint
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "last_page": current_page,
            "total_downloaded": total_downloaded,
            "chunk_index": chunk_index
        }, f)

    current_page += 1
    sleep(0.5)

# Save remainder
if buffer:
    save_chunk(buffer, chunk_index)
    print(f"Final partial chunk {chunk_index} saved with {len(buffer)} speeches.")
