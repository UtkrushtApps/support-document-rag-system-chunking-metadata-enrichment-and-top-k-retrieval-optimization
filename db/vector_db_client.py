import chromadb

class VectorDBClient:
    def __init__(self, host="chroma", port=8000, collection_name="support_articles"):
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = self.client.get_or_create_collection(collection_name)

    def upsert_documents(self, docs, embeddings, metadata_list):
        """
        Insert or update documents (chunks) to vector db
        Args:
            docs: list of strings (chunk texts)
            embeddings: list of np.arrays
            metadata_list: list of metadata dicts
        Returns:
            None
        """
        # TODO: Complete upsert logic here
        pass

    def search_top_k(self, query_embedding, k=5, filters=None):
        """
        Perform top-k vector search by cosine similarity. Optionally add filters on metadata.
        Args:
            query_embedding: np.array
            k: int k
            filters: dict
        Returns:
            List of dicts with chunk text, score, and metadata
        """
        # TODO: Complete top-k search logic
        pass
