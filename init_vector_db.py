import os
import sys
import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from config import RAW_DATA_URL, CHUNK_SIZE, CHUNK_OVERLAP, EMB_MODEL_NAME, COLLECTION_NAME
from db.vector_db_client import VectorDBClient
import nltk
nltk.download('punkt')

def download_and_parse_csv():
    print("[INFO] Downloading support dataset...")
    r = requests.get(RAW_DATA_URL)
    r.raise_for_status()
    csv_fp = "data/support_docs.csv"
    os.makedirs("data", exist_ok=True)
    with open(csv_fp, "w", encoding="utf-8") as f:
        f.write(r.text)
    df = pd.read_csv(csv_fp)
    print(f"[INFO] Loaded {df.shape[0]} support documents.")
    return df

def chunk_text(text, chunk_size, overlap):
    tokens = nltk.word_tokenize(str(text))
    results = []
    start = 0
    L = len(tokens)
    chunk_idx = 0
    while start < L:
        end = min(L, start+chunk_size)
        chunk_tokens = tokens[start:end]
        chunk_str = " ".join(chunk_tokens)
        results.append((chunk_idx, chunk_str))
        chunk_idx += 1
        start += chunk_size - overlap
    return results

def main():
    df = download_and_parse_csv()
    model = SentenceTransformer(EMB_MODEL_NAME)
    vdb = VectorDBClient()
    # Clear old collection (optional: for idempotency)
    #vdb.collection.delete()
    batch_chunks = []
    batch_metas = []
    for i, row in df.iterrows():
        chunks = chunk_text(row['text'], CHUNK_SIZE, CHUNK_OVERLAP)
        for idx, chunk in chunks:
            batch_chunks.append(chunk)
            batch_metas.append({
                "doc_id": row['id'],
                "category": row['category'],
                "priority": row['priority'],
                "date": row['date'],
                "chunk_idx": idx
            })
    print(f"[INFO] Generated {len(batch_chunks)} chunk documents.")
    print("[INFO] Embedding all chunks (batched)...")
    chunk_embeddings = model.encode(batch_chunks, show_progress_bar=True, batch_size=16)
    print("[INFO] Storing in Chroma collection...")
    vdb.upsert_documents(batch_chunks, chunk_embeddings, batch_metas)
    print("[SUCCESS] Chroma collection ready.")
    sys.exit(0)

if __name__ == '__main__':
    main()
