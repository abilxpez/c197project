import numpy as np

# Load the NPZ file
file_path = "congress_speech_embeddings_100k_combined.npz"
data = np.load(file_path, allow_pickle=True)

# Extract and display the first 10 entries
doc_ids = data["doc_ids"]
dates = data["dates"]
embeddings = data["embeddings"]

# Prepare preview of first 10 speeches
preview = []
for i in range(10):
    preview.append({
        "id": doc_ids[i],
        "date": dates[i],
        "embedding_preview": embeddings[i][:5].tolist()  # Show first 5 values of the embedding
    })

print(preview)
