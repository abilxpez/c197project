# embed_text.py

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()

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
    text = "This is a sample sentence."
    embedding = get_embedding(text)
    print(embedding)
