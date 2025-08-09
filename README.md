# Task Overview

You are tasked with optimizing a Customer Support Retrieval-Augmented Generation (RAG) system. The core infrastructure, including the Chroma vector database and word embedding model, is fully automated. However, support query results are currently poor because documents were chunked into large, non-overlapping segments and lack key metadata.

Your job is to:
- Implement improved text chunking (200 tokens per chunk with 50-token overlap)
- Attach the right metadata (category, priority, date) to each chunk
- Generate high-quality chunk embeddings using the provided sentence-transformers model
- Optimize top-k (k=5) cosine similarity retrieval for semantic search relevance
- Measure retrieval improvements using recall@k and provided queries

## Retrieval System Gaps
- Current document chunks are too large and miss contextual relationships, harming retrieval quality
- Lack of chunk overlap causes context dilution and missed matches
- Missing metadata makes filtering, ranking, and relevance checks difficult
- Retrieval function needs proper implementation and performance evaluation

**All infrastructure is already set up. You ONLY need to complete the chunking, metadata, embedding, and retrieval logic.**

## Database Access
- **Vector DB Host:** `<DROPLET_IP>`
- **Port:** 8000 (Chroma REST API)
- **Collection:** support_articles
- **Embedding Dimension:** 384 (all-MiniLM-L6-v2)
- **Chunk Metadata:** Each entry includes category, priority, date, chunk_idx, and original document reference

Explore the database via the Chroma Python client or REST API, and use metadata fields for verification or filtering.

## Objectives
- All support content is split into 200-token overlapping (50 tokens) chunks
- Each chunk is embedded and stored with metadata
- Retrieval code correctly returns the top 5 contextually relevant chunks for a query
- Results are measured with recall@5 and spot-checked for accuracy

## How to Verify
- Use the supplied FastAPI `/search` endpoint for search tests
- Try provided sample queries and confirm that the most relevant chunks are in the retrieved top-5
- Check retrieval quality manually and use recall@5 as a quantitative metric

---
**Do NOT modify infrastructure or setup scripts. Focus on improving the chunking/embedding pipeline and the core retrieval logic in the indicated code files.**