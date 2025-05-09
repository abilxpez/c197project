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

CHECKPOINT_PATH = "label_checkpoint.json"

# Topic labels
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

# Prompt with numbered labels
def build_prompt(text: str) -> str:
    numbered_topics = "\n".join([f"{i+1}. {label}" for i, label in enumerate(topic_labels)])
    return (
        f"You are a classification assistant labeling real U.S. presidential speeches based on the main topics discussed.\n\n"
        f"From the numbered list of topics below, select up to 5 that are clearly and substantially addressed in the speech.\n"
        f"- Respond ONLY with a comma-separated list of topic numbers (e.g., 3, 5, 12).\n"
        f"- Do NOT modify or write out the labels.\n"
        f"- If no topics are clearly and substantially discussed, respond with: None\n\n"
        f"Numbered Topics:\n{numbered_topics}\n\n"
        f"---\nSpeech:\n{text}"
    )

# Break text into chunks
def chunk_text(text: str, max_chars: int = 12000) -> List[str]:
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

# Convert GPT output numbers to topic labels
def classify_speech(text: str) -> List[str]:
    chunks = chunk_text(text)
    label_indices = set()

    for idx, chunk in enumerate(chunks):
        prompt = build_prompt(chunk)
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            output = response.choices[0].message.content.strip()

            if output.lower() == "none":
                continue

            for item in output.split(","):
                item = item.strip()
                if item.isdigit():
                    index = int(item) - 1
                    if 0 <= index < len(topic_labels):
                        label_indices.add(index)

        except Exception as e:
            print(f"Error on chunk {idx+1}: {e}")
        time.sleep(1)

    if not label_indices:
        return ["Unlabeled"]

    return [topic_labels[i] for i in sorted(label_indices)[:5]]

# Save checkpoint to file
def save_checkpoint(results):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(results, f)
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")

# Load from checkpoint file
def load_checkpoint():
    with open(CHECKPOINT_PATH, "r") as f:
        return json.load(f)

# Main script
if __name__ == "__main__":
    with open("speeches.json", "r") as f:
        all_documents = json.load(f)

    # Resume from checkpoint if exists
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        results = load_checkpoint()
        labeled_docs = {r["doc_name"] for r in results}
    else:
        results = []
        labeled_docs = set()

    total = len(all_documents)
    for i, doc in enumerate(all_documents, 1):
        if doc["doc_name"] in labeled_docs:
            continue

        print(f"\n[{i}/{total}] Classifying: {doc['doc_name']}")
        labels = classify_speech(doc["transcript"])
        print(f"Labels: {labels}")

        result = {
            "doc_name": doc["doc_name"],
            "date": doc["date"],
            "labels": labels
        }
        results.append(result)
        save_checkpoint(results)

    # Save final results to .npz
    np.savez("speech_labels.npz",
        doc_names=np.array([r["doc_name"] for r in results]),
        dates=np.array([r["date"] for r in results]),
        labels=np.array([r["labels"] for r in results], dtype=object))

    try:
        os.remove(CHECKPOINT_PATH)
        print("Removed checkpoint file.")
    except Exception as e:
        print(f"Could not delete checkpoint file: {e}")
