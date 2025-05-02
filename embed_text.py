import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# Load .env file for API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Token-safe truncation
def truncate_to_token_limit(text, max_tokens=8191, model="text-embedding-ada-002"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return encoding.decode(tokens)

# Get embedding from OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

# Save metadata and embeddings to .npz file
def save_metadata_and_embeddings(doc_names, dates, embeddings, out_path="speech_embeddings.npz"):
    np.savez(out_path,
             doc_names=np.array(doc_names),
             dates=np.array(dates),
             embeddings=np.array(embeddings))
    print(f"\nSaved embeddings to {out_path}")

# Main logic
if __name__ == "__main__":
    with open("speeches.json", "r") as f:
        documents = json.load(f)

    doc_names = []
    dates = []
    embeddings = []

    total = len(documents)
    for i, doc in enumerate(documents, 1):
        name = doc["doc_name"]
        date = doc["date"]
        transcript = doc["transcript"]

        print(f"[{i}/{total}] Processing document: {name}")

        truncated = truncate_to_token_limit(transcript)
        try:
            embedding = get_embedding(truncated)
            print(f"Embedding preview for '{name[:30]}...': {embedding[:5]}...")
        except Exception as e:
            print(f"Error embedding '{name}': {e}")
            continue

        doc_names.append(name)
        dates.append(date)
        embeddings.append(embedding)

    print(f"\nFinished embedding {len(embeddings)} out of {total} documents.")
    save_metadata_and_embeddings(doc_names, dates, embeddings)
