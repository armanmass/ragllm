import os
import json
from pathlib import Path

import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
EMBED_DIR = BASE_DIR / "embeddings"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

#load sentence transformer model
print('Loading sentence transformer model...')
embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)

#load faiss index
faiss_index_path = EMBED_DIR / "faiss_index.bin"
if not faiss_index_path.exists():
    raise FileNotFoundError(f"FAISS index file not found at {faiss_index_path}")
index = faiss.read_index(str(faiss_index_path))

#load metadata
meta_path = EMBED_DIR / "metadata.json"
if not meta_path.exists():
    raise FileNotFoundError(f"Metadata file not found at {meta_path}")
with open(meta_path, 'r',encoding='utf-8') as f:
    METADATA = json.load(f)

dim = index.d
print(f'Loaded FAISS index with dimension: {dim} with total {index.ntotal} vectors')

def embed_query(query):
    vec = embedder.encode(query, convert_to_numpy=True, show_progress_bar=False)
    return vec.astype(np.float32)

def top_k(query, k=5):
    q_vec = embed_query(query).reshape(1, -1)
    distances, indices = index.search(q_vec, k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(METADATA):
            results.append(METADATA[idx])
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python retriever.py \"Your question here\" [top_k]")
        sys.exit(1)
    question = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    hits = top_k(question, k)
    print(json.dumps(hits, indent=2))


