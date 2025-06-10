import os
import sys
import errno

import tiktoken
import PyPDF2
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
CHUNKS_DIR = DATA_DIR.parent / "chunks"
MAX_TOKENS = 60
LANG = 'gpt-3.5-turbo'

os.makedirs(CHUNKS_DIR, exist_ok=True)
encoding = tiktoken.encoding_for_model(LANG)

def pdf_to_text(path):
    reader = PyPDF2.PdfReader(path)
    fulltext = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            fulltext.append(text)
    return "\n".join(fulltext)

def chunk_text(text, max_tokens=MAX_TOKENS):
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens,len(tokens))
        chunk = tokens[start:end]
        chunks.append(encoding.decode(chunk))
        start = end
    return chunks

def main():
    for pdf_file in DATA_DIR.glob("*.pdf"):
        print(f"Processing {pdf_file.name}...")
        raw_text = pdf_to_text(pdf_file)

        if not raw_text.strip():
            print(f"Warning: {pdf_file.name} is empty or could not be read.")
            continue
        
        chunks = chunk_text(raw_text)
        base_name = pdf_file.stem

        for i,chunk in enumerate(chunks):
            chunk_file_path = CHUNKS_DIR / f"{base_name}_chunk_{i+1:02d}.txt"
            with open(chunk_file_path, 'w', encoding='utf-8') as f:
                f.write(chunk)
        
        print(f' > Saved {len(chunks)} chunks for {pdf_file.name}.')


if __name__ == "__main__":
    main()


