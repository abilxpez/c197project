import os
import json
import requests
from time import sleep

# Config
MIN_PER_LABEL = 50
MAX_PER_LABEL = 100
MAX_EMPTY_PAGES = 50
INPUT_FILE = "sampled_labeled_speeches.json"
OUTPUT_FILE = "congress_sampled_labeled_speeches.json"

# Load label-keyword mapping
with open("label_keywords.json", "r") as f:
    label_keywords = json.load(f)

target_labels = list(label_keywords.keys())

# Load existing speeches and counts
if os.path.exists(INPUT_FILE):
    with open(INPUT_FILE, "r") as f:
        existing_data = json.load(f)
else:
    existing_data = []

# Track existing speeches and counts
existing_ids = set()
label_counts = {label: 0 for label in target_labels}
sampled_speeches = {}

for speech in existing_data:
    speech_id = speech["id"]
    existing_ids.add(speech_id)
    sampled_speeches[speech_id] = speech
    for label in speech.get("labels", []):
        if label in label_counts:
            label_counts[label] += 1

print("Loaded existing data.")
print("Initial label counts:")
for label in label_counts:
    print(f"  {label}: {label_counts[label]}")

# Start scraping more
BASE_URL = "http://congressionalspeech.lib.uiowa.edu/api.php/speeches"
PAGE_SIZE = 50
current_page = 1
empty_page_streak = 0

print("\nAugmenting speeches...")

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

    if not speeches:
        print("No more data returned.")
        break

    new_matches_found = False

    for entry in speeches:
        speech_id = entry["id"]
        if speech_id in existing_ids:
            continue  # Skip already collected speeches

        speech_text = entry.get("speaking", "").lower()
        matched_labels = []

        for label in target_labels:
            if label_counts[label] >= MAX_PER_LABEL:
                continue
            keywords = [kw.lower() for kw in label_keywords[label]]
            if any(kw in speech_text for kw in keywords):
                matched_labels.append(label)

        if matched_labels:
            new_labels = []
            for label in matched_labels:
                if label_counts[label] < MAX_PER_LABEL:
                    new_labels.append(label)
                    label_counts[label] += 1

            if new_labels:
                sampled_speeches[speech_id] = {
                    "id": speech_id,
                    "date": entry.get("date", ""),
                    "title": entry.get("title", ""),
                    "text": entry.get("speaking", ""),
                    "speaker_state": entry.get("speaker_state", ""),
                    "speaker_party": entry.get("speaker_party", ""),
                    "labels": new_labels
                }
                existing_ids.add(speech_id)
                print(f" New match (ID {speech_id}): {entry.get('title', '')[:40]} — Labels: {new_labels}")
                new_matches_found = True

    if new_matches_found:
        empty_page_streak = 0
    else:
        empty_page_streak += 1
        print(f"  No new matches on page {current_page} (Streak: {empty_page_streak}/{MAX_EMPTY_PAGES})")

    if empty_page_streak >= MAX_EMPTY_PAGES:
        print("\n Stopping: hit 50 consecutive pages without new matches.")
        break

    current_page += 1
    sleep(0.5)

# Save updated result
final_samples = list(sampled_speeches.values())
with open(OUTPUT_FILE, "w") as f:
    json.dump(final_samples, f, indent=2)

print(f"\n Finished! Total speeches saved: {len(final_samples)}")
print("Final label counts:")
for label in target_labels:
    print(f"  {label}: {label_counts[label]}")
