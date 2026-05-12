from __future__ import annotations

import faiss
import numpy as np


class FAISSVectorStore:
    """
    Lightweight FAISS-backed vector store.
    Uses IndexFlatIP (inner product) on L2-normalized vectors -> cosine similarity.
    """

    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: list[str] = []

    def add(self, texts: list[str], embeddings: list[list[float]]) -> None:
        """
        Normalize embeddings, add to FAISS index.
        Store corresponding texts in self.metadata.
        """
        if not texts:
            return
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadata.extend(texts)

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        """
        Normalize the query embedding.
        Search FAISS for top_k nearest neighbors.
        Return a list of dicts: [{"rank": 1, "score": float, "text": str}, ...]
        """
        if len(self) == 0:
            return []
        k = min(top_k, len(self))
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            results.append({"rank": rank, "score": float(score), "text": self.metadata[idx]})
        return results

    def __len__(self) -> int:
        """Return total number of stored vectors."""
        return int(self.index.ntotal)
