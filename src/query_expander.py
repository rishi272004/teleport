from __future__ import annotations


class QueryExpander:
    """
    Mocks vertexai.generative_models.GenerativeModel for query expansion.
    In production, replace _mock_generate() with a real Vertex AI API call.
    """

    EXPANSION_MAP = {
        "peak load": "high traffic, auto-scaling, load balancing, server capacity, horizontal scaling, throughput",
        "retrieval": "semantic search, vector similarity, embedding lookup, nearest neighbor, FAISS, recall",
        "embedding": "vector representation, dense encoding, semantic similarity, sentence-transformers, floating-point vectors",
        "similarity": "cosine distance, inner product, L2 norm, angular similarity, vector comparison",
        "production": "deployment, scalability, Vertex AI, GCP, cloud infrastructure, real-time serving",
        "chunking": "text splitting, token windows, overlap, document segmentation, context preservation",
        "hallucination": "grounding, faithfulness, retrieval-augmented, factual accuracy, source attribution",
        "latency": "response time, milliseconds, throughput, performance, query speed",
        "rag": "retrieval-augmented generation, document grounding, knowledge base, vector search, context injection",
        "query expansion": "query rewriting, intent disambiguation, synonym augmentation, semantic enrichment",
    }

    def __init__(self, model_name: str = "gemini-1.0-pro"):
        self.model_name = model_name

    def expand(self, query: str) -> str:
        """
        Takes the original user query string.
        Returns an expanded query string enriched with related terms.
        Logs both the original and expanded queries.
        """
        result = self._mock_generate(query)
        print(f'[QueryExpander] Original: "{query}"')
        print(f'[QueryExpander] Expanded: "{result}"')
        return result

    def _mock_generate(self, query: str) -> str:
        """
        Deterministic expansion: scan query for known keywords (case-insensitive),
        append their expansion terms, deduplicate, return as a single string.
        If no keywords match, return the original query unchanged.
        """
        lowered = query.lower()
        expansions = []
        for keyword, expansion in self.EXPANSION_MAP.items():
            if keyword in lowered:
                expansions.append(expansion)
        if not expansions:
            return query
        combined = f"{query}, {', '.join(expansions)}"
        terms = combined.split(", ")
        deduped_terms = list(dict.fromkeys(terms))
        return ", ".join(deduped_terms)
