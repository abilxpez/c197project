import os
import json
import time
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHECKPOINT_FILE = "embedding_checkpoint.json"
OUTPUT_JSON = "congress_speeches_labeled_embedded.json"
OUTPUT_NPZ = "congress_speeches_labeled_embedded.npz"

# Truncate text to token limit
def truncate_to_token_limit(text, max_tokens=8191, model="text-embedding-3-small"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return encoding.decode(tokens)

# Get embedding
def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# Load existing checkpoint
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return []

# Save intermediate results
def save_checkpoint(results):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(results, f)
    print(f"Checkpoint saved with {len(results)} speeches.")

# MAIN
if __name__ == "__main__":
    with open("congress_speeches_labeled.json", "r") as f:
        all_speeches = json.load(f)

    results = load_checkpoint()
    embedded_ids = set([entry["id"] for entry in results])
    total = len(all_speeches)

    for i, doc in enumerate(all_speeches, 1):
        doc_id = doc["id"]
        if doc_id in embedded_ids:
            continue

        print(f"[{i}/{total}] Embedding ID {doc_id} — {doc['title'][:50]}...")
        try:
            truncated = truncate_to_token_limit(doc["text"])
            embedding = get_embedding(truncated)
        except Exception as e:
            print(f"Error on ID {doc_id}: {e}")
            continue

        doc["embedding"] = embedding
        results.append(doc)

        if len(results) % 50 == 0:
            save_checkpoint(results)
            time.sleep(1)

    # Final save
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f" Saved full embedded data to {OUTPUT_JSON}")

    # Save .npz version
    np.savez(
        OUTPUT_NPZ,
        ids=np.array([d["id"] for d in results]),
        dates=np.array([d["date"] for d in results]),
        texts=np.array([d["text"] for d in results]),
        labels=np.array([d["labels"] for d in results], dtype=object),
        embeddings=np.array([d["embedding"] for d in results]),
        states=np.array([d["speaker_state"] for d in results]),
        parties=np.array([d["speaker_party"] for d in results])
    )
    print(f" Saved final embeddings to {OUTPUT_NPZ}")

    # Cleanup checkpoint
    try:
        os.remove(CHECKPOINT_FILE)
    except Exception as e:
        print(f"Could not remove checkpoint: {e}")
