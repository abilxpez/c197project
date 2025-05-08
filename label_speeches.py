import os
import time
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from typing import List

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load speeches
with open("speeches.json", "r") as f:
    all_documents = json.load(f)

documents = all_documents[:5]  # For testing; change to all_documents for full run

topic_labels = [
    "Economy and Trade",
    "National Defense and Military",
    "Foreign Policy and Diplomacy",
    "Immigration and Border Policy",
    "Civil Rights and Racial Equality",
    "Women's Rights",
    "LGBTQ+ Rights",
    "Law Enforcement and Criminal Justice",
    "Healthcare and Public Health",
    "Education and Schools",
    "Science, Technology, and Innovation",
    "Climate and Environment",
    "Infrastructure and Transportation",
    "Government Reform and Corruption",
    "Elections and Democratic Institutions",
    "Religion, Values, and National Identity",
    "Social Welfare and Poverty",
    "Labor, Jobs, and Workers' Rights",
    "Gun Policy and Second Amendment",
    "Energy and Natural Resources",
    "Terrorism, Homeland Security, and War on Terror",
    "Indigenous and Tribal Affairs"
]

def build_prompt(text: str) -> str:
    topics = ", ".join(topic_labels)
    return (
        f"You are a classification assistant labeling real U.S. presidential speeches based on the main topics discussed.\n\n"
        f"From the list of topics below, select up to 5 that are clearly and substantially addressed in the speech.\n"
        f"- Do not guess or infer hidden meanings.\n"
        f"- Only choose a topic if it is explicitly or repeatedly discussed.\n"
        f"- Ignore minor references or passing mentions.\n\n"
        f"Respond with a comma-separated list of only the chosen topics, no explanations or extra text.\n\n"
        f"Topics:\n{topics}\n\n"
        f"---\nSpeech:\n{text}"
    )

def chunk_text(text: str, max_chars: int = 12000) -> List[str]:
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def classify_speech(text: str) -> List[str]:
    chunks = chunk_text(text)
    labels = set()

    for idx, chunk in enumerate(chunks):
        prompt = build_prompt(chunk)
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            output = response.choices[0].message.content
            chunk_labels = [label.strip() for label in output.split(",")]
            labels.update(chunk_labels)
        except Exception as e:
            print(f"Error on chunk {idx+1}: {e}")
        time.sleep(1)
        
    return list(labels)[:5]

# Run
results = []
for doc in documents:
    print(f"\nClassifying: {doc['doc_name']}")
    labels = classify_speech(doc["transcript"])
    print(f"Labels: {labels}")
    results.append({
        "doc_name": doc["doc_name"],
        "date": doc["date"],
        "labels": labels
    })

# Save
np.savez("speech_labels_test.npz",
    doc_names=np.array([r["doc_name"] for r in results]),
    dates=np.array([r["date"] for r in results]),
    labels=np.array([r["labels"] for r in results], dtype=object))
