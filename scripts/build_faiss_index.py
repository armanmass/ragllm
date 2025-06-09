import os
import json

from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import numpy as np
import faiss
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
CHUNK_DIR = BASE_DIR / 'chunks'
EMBED_DIR = BASE_DIR / 'embeddings'
os.makedirs(EMBED_DIR, exist_ok=True)

if torch.cuda.is_available():
    print("Using GPU for embedding generation")
    device = 'cuda'
else:
    print("Using CPU for embedding generation")
    device = 'cpu'


EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'
print(f'Loading embedding model: {EMBED_MODEL_NAME} on device: {device}')
embedder = SentenceTransformer(EMBED_MODEL_NAME, device=device)

chunk_files = list(CHUNK_DIR.glob('*.txt'))
print(f'Found {len(chunk_files)} chunk files to process.')

all_embeddings = []
metadata = []

for file_path in tqdm(chunk_files, desc='Embedding chunks'):
    text = file_path.read_text(encoding='utf-8').strip()

    if not text: continue

    embedding = embedder.encode(text, convert_to_numpy=True, show_progress_bar=False)
    all_embeddings.append(embedding)

    metadata.append({
        'chunk_filepath': str(file_path),
        'preview': text.replace('\n',' ')[:200] + ('...' if len(text) > 100 else '')
    })

emb_matrix = np.vstack(all_embeddings)
dim = emb_matrix.shape[1]
print(f'Embedding dimension: {dim}. Total vectors: {len(all_embeddings)}')

index = faiss.IndexFlatL2(dim)
index.add(emb_matrix)

faiss_index_path = EMBED_DIR / 'faiss_index.bin'
faiss.write_index(index, str(faiss_index_path))

with open(EMBED_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print(f'FAISS index saved to: {faiss_index_path}')
print(f'Metadata saved to: {EMBED_DIR / "metadata.json"}')
print(f'Metadata contains {len(metadata)} entries.')
print('Embedding and indexing complete.')