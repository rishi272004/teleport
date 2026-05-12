import numpy as np
import pytest

from src.vector_store import FAISSVectorStore


def test_add_and_len():
    store = FAISSVectorStore(384)
    texts = ["doc one", "doc two", "doc three"]
    embeddings = np.random.rand(3, 384)
    store.add(texts, embeddings)
    assert len(store) == 3


def test_search_returns_top_k():
    store = FAISSVectorStore(384)
    texts = [f"doc {i}" for i in range(5)]
    embeddings = np.random.rand(5, 384)
    store.add(texts, embeddings)

    results = store.search(np.random.rand(384), top_k=3)
    assert len(results) == 3


def test_search_result_schema():
    store = FAISSVectorStore(384)
    texts = [f"doc {i}" for i in range(3)]
    embeddings = np.random.rand(3, 384)
    store.add(texts, embeddings)

    results = store.search(np.random.rand(384), top_k=3)
    for result in results:
        assert set(result.keys()) == {"rank", "score", "text"}


def test_cosine_identical_vector():
    store = FAISSVectorStore(384)
    vector = np.random.rand(384)
    store.add(["doc one"], [vector])

    results = store.search(vector, top_k=1)
    assert results[0]["score"] == pytest.approx(1.0, abs=0.01)


def test_top_k_clamping():
    store = FAISSVectorStore(384)
    texts = ["doc one", "doc two"]
    embeddings = np.random.rand(2, 384)
    store.add(texts, embeddings)

    results = store.search(np.random.rand(384), top_k=5)
    assert len(results) == 2
