import os
import json
import numpy as np

CHECKPOINT_FILE = "congress_checkpoint.json"
RECOVERY_PROGRESS = "recovery_progress_checkpoint.json"
CHUNK_SIZE = 50000

def extract_and_save_chunks(filepath, already_recovered=0, chunk_index=0):
    with open(filepath, "r") as f:
        raw = f.read()

    start = raw.find('"data": [')
    if start == -1:
        raise ValueError("Could not find 'data' array.")
    start_bracket = raw.find('[', start)
    content = raw[start_bracket + 1:]

    brace_depth = 0
    in_string = False
    escape = False
    obj_start = None

    obj_count = 0
    i = 0
    buffer = []

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
                    buffer.append(obj)
                    obj_count += 1

                    if obj_count % 50 == 0:
                        print(f"[{obj.get('id', '?')}] {obj.get('title', '')[:50]} — {' '.join(obj.get('text', '').split()[:20])}...\n")

                    if len(buffer) == CHUNK_SIZE:
                        save_chunk(buffer, chunk_index)
                        buffer = []
                        chunk_index += 1

                        with open(RECOVERY_PROGRESS, "w") as f:
                            json.dump({"count": obj_count, "chunk_index": chunk_index}, f)
                        print(f"Saved chunk {chunk_index}, total recovered: {obj_count}")

                except json.JSONDecodeError:
                    pass
        i += 1

    # Save remaining buffer
    if buffer:
        save_chunk(buffer, chunk_index)
        with open(RECOVERY_PROGRESS, "w") as f:
            json.dump({"count": obj_count, "chunk_index": chunk_index + 1}, f)
        print(f"Final chunk {chunk_index} saved with {len(buffer)} speeches.")

def save_chunk(data, index):
    suffix = str(index + 1).zfill(5)
    json_path = f"recovered_part_{suffix}.json"
    npz_path = f"recovered_part_{suffix}.npz"

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    np.savez(
        npz_path,
        ids=np.array([d["id"] for d in data]),
        dates=np.array([d["date"] for d in data]),
        titles=np.array([d["title"] for d in data]),
        texts=np.array([d["text"] for d in data]),
        speaker_states=np.array([d["speaker_state"] for d in data]),
        speaker_parties=np.array([d["speaker_party"] for d in data]),
    )
    print(f"Saved to {json_path} and {npz_path}")

# Load progress
if os.path.exists(RECOVERY_PROGRESS):
    with open(RECOVERY_PROGRESS, "r") as f:
        prog = json.load(f)
    already_recovered = prog.get("count", 0)
    chunk_index = prog.get("chunk_index", 0)
    print(f"Resuming from speech #{already_recovered}, chunk {chunk_index}")
else:
    already_recovered = 0
    chunk_index = 0
    print("Starting fresh recovery...")

extract_and_save_chunks(CHECKPOINT_FILE, already_recovered, chunk_index)
