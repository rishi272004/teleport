from __future__ import annotations

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wraps sentence-transformers to simulate Vertex AI textembedding-gecko.
    The .get_embeddings() method mirrors the Vertex AI SDK interface.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Accepts a list of strings.
        Returns a list of embedding vectors (list of floats).
        Mirrors the Vertex AI TextEmbeddingModel.get_embeddings() signature.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Convenience method for embedding a single query string."""
        return self.get_embeddings([text])[0]
