import os
import json
import requests
from time import sleep
from collections import defaultdict
from datetime import datetime

# Config
PER_BIN_LIMIT = 250
MAX_PER_LABEL = 1000
INPUT_FILE = "congress_sampled_unlabeled_speeches_updated.json"
OUTPUT_FILE = "congress_sampled_unlabeled_speeches_updated2.json"
CHECKPOINT_INTERVAL = 50  # Save after every 50 new speeches

# Load label-keyword mapping
with open("label_keywords.json", "r") as f:
    label_keywords = json.load(f)

target_labels = list(label_keywords.keys())

# Tracking
existing_ids = set()
label_counts = {label: 0 for label in target_labels}
label_timebin_counts = {label: defaultdict(int) for label in target_labels}
sampled_speeches = {}
newly_added = 0  # For checkpointing

def get_5yr_bin(date_str):
    try:
        year = int(date_str[:4])
        return (year // 5) * 5
    except:
        return None

# Load existing data
if os.path.exists(INPUT_FILE):
    with open(INPUT_FILE, "r") as f:
        existing_data = json.load(f)
else:
    existing_data = []

for speech in existing_data:
    speech_id = speech["id"]
    existing_ids.add(speech_id)
    sampled_speeches[speech_id] = speech
    for label in speech.get("labels", []):
        if label in label_counts:
            label_counts[label] += 1
            date_str = speech.get("date", "")
            time_bin = get_5yr_bin(date_str)
            if time_bin is not None:
                label_timebin_counts[label][time_bin] += 1

print("Loaded existing data.")
print("Initial label counts:")
for label in label_counts:
    print(f"  {label}: {label_counts[label]}")

# Start scraping
BASE_URL = "http://congressionalspeech.lib.uiowa.edu/api.php/speeches"
PAGE_SIZE = 50
current_page = 1

print("\nAugmenting speeches...")

try:
    while any(label_counts[label] < MAX_PER_LABEL for label in target_labels):
        print(f"\nFetching page {current_page}...")

        url = f"{BASE_URL}?transform=1&order=id&page={current_page},{PAGE_SIZE}"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Request failed: {response.status_code}")
            break

        speeches = response.json()
        if isinstance(speeches, dict) and "speeches" in speeches:
            speeches = speeches["speeches"]
        elif not isinstance(speeches, list):
            print("Unexpected response format.")
            break

        for entry in speeches:
            speech_id = entry["id"]
            if speech_id in existing_ids:
                continue

            speech_text = entry.get("speaking", "").lower()
            speech_date = entry.get("date", "")
            time_bin = get_5yr_bin(speech_date)
            if time_bin is None:
                continue

            matched_labels = []
            for label in target_labels:
                if label_counts[label] >= MAX_PER_LABEL:
                    continue
                keywords = [kw.lower() for kw in label_keywords[label]]
                if any(kw in speech_text for kw in keywords):
                    if label_timebin_counts[label][time_bin] < PER_BIN_LIMIT:
                        matched_labels.append(label)

            if matched_labels:
                new_labels = []
                for label in matched_labels:
                    if (label_counts[label] < MAX_PER_LABEL and 
                        label_timebin_counts[label][time_bin] < PER_BIN_LIMIT):
                        
                        new_labels.append(label)
                        label_counts[label] += 1
                        label_timebin_counts[label][time_bin] += 1

                if new_labels:
                    sampled_speeches[speech_id] = {
                        "id": speech_id,
                        "date": speech_date,
                        "title": entry.get("title", ""),
                        "text": entry.get("speaking", ""),
                        "speaker_state": entry.get("speaker_state", ""),
                        "speaker_party": entry.get("speaker_party", ""),
                        "labels": new_labels
                    }
                    existing_ids.add(speech_id)
                    newly_added += 1
                    print(f" New match (ID {speech_id}): {entry.get('title', '')[:40]} — Labels: {new_labels}")

                    # Checkpoint every 50 additions
                    if newly_added % CHECKPOINT_INTERVAL == 0:
                        with open(OUTPUT_FILE, "w") as f:
                            json.dump(list(sampled_speeches.values()), f, indent=2)
                        print(f"Checkpoint: {newly_added} new speeches saved so far.")

        current_page += 1
        sleep(0.5)

except KeyboardInterrupt:
    print("\nKeyboardInterrupt detected — saving progress...")

finally:
    with open(OUTPUT_FILE, "w") as f:
        json.dump(list(sampled_speeches.values()), f, indent=2)
    print(f"\nFinal save completed. Total speeches saved: {len(sampled_speeches)}")
    print("Final label counts:")
    for label in target_labels:
        print(f"  {label}: {label_counts[label]}")
