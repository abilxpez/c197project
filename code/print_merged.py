import json

# Load the JSON data
file_path = "merged_labeled_speeches_umap_pca.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Print first 5 entries with selected fields
for i, entry in enumerate(data[:5], 1):
    print(f"\n=== Entry {i} ===")
    print(f"ID: {entry.get('id')}")
    print(f"Date: {entry.get('date')}")
    print(f"Speaker State: {entry.get('speaker_state')}")
    print(f"Speaker Party: {entry.get('speaker_party')}")
    print(f"Labels: {entry.get('labels')}")
    
    embedding_preview = entry.get("embedding", [])[:5]
    print(f"Embedding (first 5): {embedding_preview}")
    
    print(f"PCA: {entry.get('pca')}")
    print(f"UMAP: {entry.get('umap')}")
