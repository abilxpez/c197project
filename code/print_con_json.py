import json
from collections import Counter

CHECKPOINT_PATH = "model_labeled_speeches_44K.json"

# Load checkpointed results
with open(CHECKPOINT_PATH, "r") as f:
    speeches = json.load(f)

label_counter = Counter()
unlabeled_count = 0
total_labeled = 0

for speech in speeches:
    labels = speech.get("labels", [])
    if labels == ["Unlabeled"]:
        unlabeled_count += 1
    else:
        total_labeled += 1
        label_counter.update(labels)

print(f"\nTotal speeches in checkpoint: {len(speeches)}")
print(f" Labeled speeches: {total_labeled}")
print(f" Unlabeled speeches: {unlabeled_count}\n")

print(" Label counts:")
for label, count in label_counter.most_common():
    print(f"  {label}: {count}")
