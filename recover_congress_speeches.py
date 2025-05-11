import os
import json
import numpy as np

CHECKPOINT_FILE = "congress_checkpoint.json"
RECOVERY_JSON = "congress_speeches_recovered.json"
RECOVERY_NPZ = "congress_speeches_recovered.npz"
RECOVERY_PROGRESS = "recovery_progress_checkpoint.json"

def extract_all_by_brace_balance(filepath, already_recovered=0):
    with open(filepath, "r") as f:
        raw = f.read()

    start = raw.find('"data": [')
    if start == -1:
        raise ValueError("Could not find 'data' array.")
    start_bracket = raw.find('[', start)
    if start_bracket == -1:
        raise ValueError("Could not find '[' after 'data':")

    content = raw[start_bracket + 1:]

    objects = []
    brace_depth = 0
    in_string = False
    escape = False
    obj_start = None

    obj_count = 0
    i = 0
    while i < len(content):
        char = content[i]

        if char == '"' and not escape:
            in_string = not in_string
        elif char == '\\' and not escape:
            escape = True
            i += 1
            continue
        escape = False

        if in_string:
            i += 1
            continue

        if char == '{':
            if brace_depth == 0:
                obj_start = i
            brace_depth += 1
        elif char == '}':
            brace_depth -= 1
            if brace_depth == 0 and obj_start is not None:
                if obj_count < already_recovered:
                    obj_count += 1
                    i += 1
                    continue
                obj_str = content[obj_start:i+1]
                try:
                    obj = json.loads(obj_str)
                    objects.append(obj)
                    obj_count += 1

                    if obj_count % 50 == 0:
                        print(f"[{obj.get('id', '?')}] {obj.get('title', '')[:50]} — {' '.join(obj.get('text', '').split()[:20])}...\n")

                    if obj_count % 100 == 0:
                        with open(RECOVERY_PROGRESS, "w") as f:
                            json.dump({
                                "count": obj_count,
                                "data": objects
                            }, f)
                        print(f"Checkpoint saved at {obj_count} recovered speeches...")

                except json.JSONDecodeError:
                    pass
        i += 1

    return objects

def save_outputs(recovered_data):
    with open(RECOVERY_JSON, "w") as f:
        json.dump(recovered_data, f, indent=2)

    np.savez(
        RECOVERY_NPZ,
        ids=np.array([d["id"] for d in recovered_data]),
        dates=np.array([d["date"] for d in recovered_data]),
        titles=np.array([d["title"] for d in recovered_data]),
        texts=np.array([d["text"] for d in recovered_data]),
        speaker_states=np.array([d["speaker_state"] for d in recovered_data]),
        speaker_parties=np.array([d["speaker_party"] for d in recovered_data]),
    )

    print(f"\nSaved final JSON to '{RECOVERY_JSON}' and NPZ to '{RECOVERY_NPZ}'")

# Load progress if it exists
if os.path.exists(RECOVERY_PROGRESS):
    with open(RECOVERY_PROGRESS, "r") as f:
        progress_data = json.load(f)
    recovered_data = progress_data["data"]
    already_recovered = progress_data["count"]
    print(f"Resuming from previously recovered {already_recovered} speeches...")
else:
    recovered_data = []
    already_recovered = 0
    print("Starting fresh recovery...")

# Continue recovery
new_data = extract_all_by_brace_balance(CHECKPOINT_FILE, already_recovered=already_recovered)
recovered_data += new_data

# Save final outputs
save_outputs(recovered_data)
