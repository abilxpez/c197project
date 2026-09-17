import numpy as np
import glob

# Set the pattern for all your chunk files
chunk_files = sorted(glob.glob("congress_speech_embeddings_chunk_*.npz"))

all_doc_ids = []
all_dates = []
all_embeddings = []

print(f"Found {len(chunk_files)} chunk files.")

for path in chunk_files:
    print(f"Loading {path}...")
    data = np.load(path, allow_pickle=True)
    
    all_doc_ids.extend(data["doc_ids"])
    all_dates.extend(data["dates"])
    all_embeddings.extend(data["embeddings"])

print(f"Total combined entries: {len(all_doc_ids)}")

# Save combined arrays into a single .npz
np.savez("congress_speech_embeddings_100k_combined.npz",
         doc_ids=np.array(all_doc_ids),
         dates=np.array(all_dates),
         embeddings=np.array(all_embeddings))

print(" Saved combined embeddings to 'congress_speech_embeddings_100k_combined.npz'")
