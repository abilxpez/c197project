import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Truncate to token limit
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

# Save final output
def save_metadata_and_embeddings(ids, dates, embeddings, out_path="congress_speech_embeddings_test.npz"):
    np.savez(out_path,
             ids=np.array(ids),
             dates=np.array(dates),
             embeddings=np.array(embeddings))
    print(f"\nSaved embeddings to {out_path}")

# Main script
if __name__ == "__main__":
    with open("congress_speeches.json", "r") as f:
        documents = json.load(f)

    documents = documents[:5]  # Only take first 5 for testing

    ids = []
    dates = []
    embeddings = []

    total = len(documents)
    for i, doc in enumerate(documents, 1):
        doc_id = doc["id"]
        date = doc["date"]
        text = doc["text"]

        print(f"[{i}/{total}] Processing document ID: {doc_id}")
        truncated = truncate_to_token_limit(text)

        try:
            embedding = get_embedding(truncated)
            print(f"Embedding preview for ID {doc_id}: {embedding[:5]}...")
        except Exception as e:
            print(f"Error embedding ID {doc_id}: {e}")
            continue

        ids.append(doc_id)
        dates.append(date)
        embeddings.append(embedding)

    print(f"\nFinished embedding {len(embeddings)} out of {total} documents.")
    save_metadata_and_embeddings(ids, dates, embeddings)
