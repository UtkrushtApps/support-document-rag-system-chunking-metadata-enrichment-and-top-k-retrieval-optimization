#!/bin/bash
set -e

echo "[INFO] Starting Chroma vector database and FastAPI services..."
docker-compose up -d
sleep 13

echo "[INFO] Running dataset fetching, chunking, and embedding job..."
docker-compose exec api python init_vector_db.py

if [ $? -eq 0 ]; then
  echo "[SUCCESS] Vector DB initialized, embeddings and metadata stored. Proceed to implement and test retrieval logic."
else
  echo "[ERROR] Initialization failed. Check logs for more details."
  exit 1
fi
