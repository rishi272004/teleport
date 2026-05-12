import numpy as np

from data.corpus import CORPUS
from src.pipeline import RAGPipeline


def test_ingest_populates_store():
    pipeline = RAGPipeline()
    pipeline.ingest(CORPUS)
    assert len(pipeline.vector_store) == 10


def test_benchmark_output_schema():
    pipeline = RAGPipeline()
    pipeline.ingest(CORPUS)

    results = pipeline.benchmark(["test query"])
    assert isinstance(results, list)
    assert len(results) == 1
    assert set(results[0].keys()) == {"query", "strategy_a", "strategy_b", "expanded_query"}


def test_benchmark_multiple_queries():
    pipeline = RAGPipeline()
    pipeline.ingest(CORPUS)

    queries = ["query one", "query two", "query three"]
    results = pipeline.benchmark(queries)
    assert len(results) == 3


def test_mock_gcp_embedding_model(mocker):
    mock_model = mocker.Mock()

    def _mock_encode(texts, convert_to_numpy=True):
        return np.ones((len(texts), 384))

    mock_model.encode.side_effect = _mock_encode
    mocker.patch("src.embedder.SentenceTransformer", return_value=mock_model)

    pipeline = RAGPipeline()
    pipeline.ingest(CORPUS)
    results = pipeline.benchmark(["test query"], top_k=3)

    assert isinstance(results, list)
