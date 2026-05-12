from __future__ import annotations

from src.embedder import EmbeddingModel
from src.query_expander import QueryExpander
from src.vector_store import FAISSVectorStore


class RAGRetriever:
    """
    Exposes Strategy A (raw vector search) and Strategy B (query-expanded search).
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedder: EmbeddingModel,
        query_expander: QueryExpander,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.query_expander = query_expander

    def retrieve_strategy_a(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Strategy A - Raw Vector Search:
        1. Embed the original query directly.
        2. Search the vector store.
        3. Return top_k results.
        """
        embedding = self.embedder.embed_single(query)
        return self.vector_store.search(embedding, top_k=top_k)

    def retrieve_strategy_b(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Strategy B - AI-Enhanced Retrieval:
        1. Expand the query using QueryExpander.
        2. Embed the expanded query.
        3. Search the vector store.
        4. Return top_k results.
        """
        expanded_query = self.query_expander.expand(query)
        embedding = self.embedder.embed_single(expanded_query)
        return self.vector_store.search(embedding, top_k=top_k)
