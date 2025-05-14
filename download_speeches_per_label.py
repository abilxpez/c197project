import os
import json
import requests
from time import sleep

# Load label-keyword mapping (first 3 labels only)
with open("label_keywords.json", "r") as f:
    label_keywords = json.load(f)

target_labels = list(label_keywords.keys())[:3]  # First 3 labels
target_count_per_label = 5
label_samples = {label: [] for label in target_labels}

# Set up API parameters
BASE_URL = "http://congressionalspeech.lib.uiowa.edu/api.php/speeches"
PAGE_SIZE = 50
current_page = 1

print("Starting sampling...")

while any(len(samples) < target_count_per_label for samples in label_samples.values()):
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

    for entry in speeches:
        speech_text = entry.get("speaking", "").lower()

        for label in target_labels:
            if len(label_samples[label]) >= target_count_per_label:
                continue

            keywords = [kw.lower() for kw in label_keywords[label]]
            if any(kw in speech_text for kw in keywords):
                sample = {
                    "label": label,
                    "id": entry["id"],
                    "date": entry.get("date", ""),
                    "title": entry.get("title", ""),
                    "text": entry.get("speaking", ""),
                    "speaker_state": entry.get("speaker_state", ""),
                    "speaker_party": entry.get("speaker_party", "")
                }
                label_samples[label].append(sample)
                print(f"✅ Match for '{label}' (ID {sample['id']}): {sample['title'][:40]}...")

    current_page += 1
    sleep(0.5)

# Combine and save
all_samples = [speech for speeches in label_samples.values() for speech in speeches]
with open("sampled_labeled_speeches.json", "w") as f:
    json.dump(all_samples, f, indent=2)

print(f"\n✅ Finished! Saved {len(all_samples)} speeches to 'sampled_labeled_speeches.json'")
