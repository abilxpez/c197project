# embed_text.py

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

# Load .env file
load_dotenv()

# Truncate to max_tokens using tokenization
def truncate_to_token_limit(text, max_tokens=8191, model="text-embedding-ada-002"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return encoding.decode(tokens)

# Create OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        input=[text],  # input must be a list, even if it's one sentence
        model=model
    )
    embedding = response.data[0].embedding
    return embedding

if __name__ == "__main__":
    # read JSON file 
    with open('speeches.json', 'r') as f:
        documents = json.load(f)  # this will be a list of dicts

    # loop through each document
    for doc in documents:
        doc_name = doc['doc_name']
        transcript = doc['transcript']

        print(f"Processing document: {doc_name}")

        # get embedding
        truncated = truncate_to_token_limit(transcript)
        embedding = get_embedding(truncated)
        # embedding = get_embedding(transcript)

        # print embedding
        print(f"Embedding for {doc_name[:30]}...: {embedding[:5]}...") 