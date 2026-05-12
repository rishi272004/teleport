import numpy as np

from src.embedder import EmbeddingModel


def test_embed_single_returns_list():
    model = EmbeddingModel()
    result = model.embed_single("hello world")
    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(value, float) for value in result)


def test_get_embeddings_batch():
    model = EmbeddingModel()
    result = model.get_embeddings(["text one", "text two"])
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(len(item) == 384 for item in result)


def test_mock_vertex_ai_interface(mocker):
    mock_model = mocker.Mock()
    mock_model.encode.return_value = np.ones((1, 384))
    mocker.patch("src.embedder.SentenceTransformer", return_value=mock_model)

    model = EmbeddingModel()
    result = model.get_embeddings(["test"])

    assert isinstance(result, list)
    assert len(result) == 1
    assert len(result[0]) == 384
