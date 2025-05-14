import os
import json
import requests
from time import sleep

# Load all 22 labels and keywords
with open("label_keywords.json", "r") as f:
    label_keywords = json.load(f)

target_labels = list(label_keywords.keys())
target_count_per_label = 50

# Track how many speeches we’ve matched per label
label_counts = {label: 0 for label in target_labels}
sampled_speeches = {}  # key: speech_id → speech info + labels

# Set up API parameters
BASE_URL = "http://congressionalspeech.lib.uiowa.edu/api.php/speeches"
PAGE_SIZE = 50
current_page = 1

print("Starting multi-label sampling for 22 labels × 50 speeches...")

while any(count < target_count_per_label for count in label_counts.values()):
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
        speech_id = entry["id"]
        speech_text = entry.get("speaking", "").lower()

        matched_labels = []
        for label in target_labels:
            if label_counts[label] >= target_count_per_label:
                continue

            keywords = [kw.lower() for kw in label_keywords[label]]
            if any(kw in speech_text for kw in keywords):
                matched_labels.append(label)

        if matched_labels:
            if speech_id not in sampled_speeches:
                sampled_speeches[speech_id] = {
                    "id": speech_id,
                    "date": entry.get("date", ""),
                    "title": entry.get("title", ""),
                    "text": entry.get("speaking", ""),
                    "speaker_state": entry.get("speaker_state", ""),
                    "speaker_party": entry.get("speaker_party", ""),
                    "labels": []
                }

            for label in matched_labels:
                if (
                    label_counts[label] < target_count_per_label and
                    label not in sampled_speeches[speech_id]["labels"]
                ):
                    sampled_speeches[speech_id]["labels"].append(label)
                    label_counts[label] += 1
                    print(f" Match for '{label}' (ID {speech_id}): {entry.get('title', '')[:40]}...")

    current_page += 1
    sleep(0.5)

# Save final result
final_samples = list(sampled_speeches.values())
with open("sampled_labeled_speeches.json", "w") as f:
    json.dump(final_samples, f, indent=2)

print(f"\n Finished! Saved {len(final_samples)} speeches with multi-labels to 'sampled_labeled_speeches.json'")
