import os
import json

CHECKPOINT_FILE = "congress_checkpoint.json"
OUTPUT_JSON = "test_first_5_recovered.json"

def extract_first_five_by_brace_balance(filepath):
    with open(filepath, "r") as f:
        raw = f.read()

    # Find the start of the "data": [
    start = raw.find('"data": [')
    if start == -1:
        raise ValueError("Could not find 'data' array.")

    # Find the actual start of the array
    start_bracket = raw.find('[', start)
    if start_bracket == -1:
        raise ValueError("Could not find '[' after 'data':")

    content = raw[start_bracket + 1:]  # skip the opening [

    objects = []
    brace_depth = 0
    in_string = False
    escape = False
    obj_start = None

    for i, char in enumerate(content):
        if char == '"' and not escape:
            in_string = not in_string
        elif char == '\\' and not escape:
            escape = True
            continue
        escape = False

        if in_string:
            continue

        if char == '{':
            if brace_depth == 0:
                obj_start = i
            brace_depth += 1
        elif char == '}':
            brace_depth -= 1
            if brace_depth == 0 and obj_start is not None:
                obj_str = content[obj_start:i+1]
                try:
                    obj = json.loads(obj_str)
                    objects.append(obj)

                    # Print preview
                    title = obj.get("title", "")[:50]
                    text_preview = " ".join(obj.get("text", "").split()[:20])
                    print(f"[{obj.get('id', '?')}] {title} — {text_preview}...\n")

                    if len(objects) == 5:
                        break
                except json.JSONDecodeError:
                    continue

    return objects

if __name__ == "__main__":
    speeches = extract_first_five_by_brace_balance(CHECKPOINT_FILE)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(speeches, f, indent=2)

    print(f"\nSaved {len(speeches)} recovered speeches to '{OUTPUT_JSON}'")
