import os
import json
import numpy as np
import random
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Token-safe truncation
def truncate_to_token_limit(text, max_tokens=8191, model="text-embedding-3-small"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return encoding.decode(tokens)

# Get embedding from OpenAI
def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# Save current progress
def save_metadata_and_embeddings(doc_ids, dates, embeddings, out_path):
    np.savez(out_path,
             doc_ids=np.array(doc_ids),
             dates=np.array(dates),
             embeddings=np.array(embeddings))
    print(f"Checkpoint saved to {out_path} [{len(doc_ids)} documents]")

# Main logic
if __name__ == "__main__":
    with open("congress_speeches_recovered.json", "r") as f:
        all_documents = json.load(f)

    print(f"Total speeches available: {len(all_documents)}")
    sample_size = 100_000
    sampled_docs = random.sample(all_documents, sample_size)
    print(f"Randomly sampled {sample_size} congressional speeches.")

    doc_ids = []
    dates = []
    embeddings = []

    checkpoint_path = "congress_speech_embeddings_checkpoint.npz"
    final_path = "congress_speech_embeddings_100k.npz"

    for i, doc in enumerate(sampled_docs, 1):
        doc_id = doc["id"]
        date = doc["date"]
        text = doc["text"]

        print(f"[{i}/{sample_size}] Processing document ID: {doc_id}")

        truncated = truncate_to_token_limit(text)
        try:
            embedding = get_embedding(truncated)
            print(f"Embedding preview for ID {doc_id}: {embedding[:5]}...")
        except Exception as e:
            print(f"Error embedding document ID {doc_id}: {e}")
            continue

        doc_ids.append(doc_id)
        dates.append(date)
        embeddings.append(embedding)

        # Save checkpoint every 1000 entries
        if i % 1000 == 0:
            save_metadata_and_embeddings(doc_ids, dates, embeddings, checkpoint_path)

    # Final save after loop
    print(f"\nFinished embedding {len(embeddings)} out of {sample_size} documents.")
    save_metadata_and_embeddings(doc_ids, dates, embeddings, final_path)
