from __future__ import annotations

from src.embedder import EmbeddingModel
from src.query_expander import QueryExpander
from src.retriever import RAGRetriever
from src.vector_store import FAISSVectorStore


class RAGPipeline:
    """
    Orchestrates ingestion and retrieval for the RAG benchmark.
    """

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = EmbeddingModel(embedding_model_name)
        self.vector_store = FAISSVectorStore(dimension=384)
        self.query_expander = QueryExpander()
        self.retriever = RAGRetriever(self.vector_store, self.embedder, self.query_expander)

    def ingest(self, corpus: list[str]) -> None:
        """
        Embeds all corpus texts and adds them to the vector store.
        Prints: "[Pipeline] Ingested N documents."
        """
        embeddings = self.embedder.get_embeddings(corpus)
        self.vector_store.add(corpus, embeddings)
        print(f"[Pipeline] Ingested {len(corpus)} documents.")

    def benchmark(self, queries: list[str], top_k: int = 3) -> list[dict]:
        """
        For each query, runs both Strategy A and Strategy B.
        Returns a list of result objects (one per query):
        [
            {
                "query": str,
                "strategy_a": [{"rank": int, "score": float, "text": str}, ...],
                "strategy_b": [{"rank": int, "score": float, "text": str}, ...],
                "expanded_query": str,
            },
            ...
        ]
        """
        results = []
        for query in queries:
            expanded_query = self.query_expander.expand(query)
            strategy_a = self.retriever.retrieve_strategy_a(query, top_k=top_k)
            strategy_b = self.retriever.retrieve_strategy_b(query, top_k=top_k)
            results.append(
                {
                    "query": query,
                    "strategy_a": strategy_a,
                    "strategy_b": strategy_b,
                    "expanded_query": expanded_query,
                }
            )
        return results
