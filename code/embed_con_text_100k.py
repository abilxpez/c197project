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

# Save current chunk to a separate file
def save_chunk(doc_ids, dates, embeddings, chunk_index, base_name="congress_speech_embeddings_chunk"):
    out_path = f"{base_name}_{chunk_index}.npz"
    np.savez(out_path,
             doc_ids=np.array(doc_ids),
             dates=np.array(dates),
             embeddings=np.array(embeddings))
    print(f"Chunk {chunk_index} saved to {out_path} [{len(doc_ids)} entries]")

# Main logic
if __name__ == "__main__":
    with open("congress_speeches_recovered.json", "r") as f:
        all_documents = json.load(f)

    print(f"Total speeches available: {len(all_documents)}")
    sample_size = 100_000
    sampled_docs = random.sample(all_documents, sample_size)
    print(f"Randomly sampled {sample_size} congressional speeches.")

    chunk_size = 1_000  # Change to 5000 if desired
    doc_ids = []
    dates = []
    embeddings = []
    chunk_index = 1

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

        # Save and reset every chunk_size entries
        if i % chunk_size == 0:
            save_chunk(doc_ids, dates, embeddings, chunk_index)
            doc_ids, dates, embeddings = [], [], []  # Reset for next chunk
            chunk_index += 1

    # Save any remaining items after final chunk
    if doc_ids:
        save_chunk(doc_ids, dates, embeddings, chunk_index)

    print(f"\nFinished embedding {sample_size} documents in {chunk_index} chunks.")
