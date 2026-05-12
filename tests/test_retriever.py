from src.embedder import EmbeddingModel
from src.query_expander import QueryExpander
from src.retriever import RAGRetriever
from src.vector_store import FAISSVectorStore
from data.corpus import CORPUS


def test_strategy_a_calls_embedder(mocker):
    embedder = mocker.Mock()
    embedder.embed_single.return_value = [0.0] * 384
    query_expander = mocker.Mock()
    vector_store = mocker.Mock()
    vector_store.search.return_value = []

    retriever = RAGRetriever(vector_store, embedder, query_expander)
    retriever.retrieve_strategy_a("test query")

    embedder.embed_single.assert_called_once_with("test query")


def test_strategy_b_calls_expander(mocker):
    embedder = mocker.Mock()
    embedder.embed_single.return_value = [0.0] * 384
    query_expander = mocker.Mock()
    query_expander.expand.return_value = "expanded"
    vector_store = mocker.Mock()
    vector_store.search.return_value = []

    retriever = RAGRetriever(vector_store, embedder, query_expander)
    retriever.retrieve_strategy_b("test query")

    query_expander.expand.assert_called_once_with("test query")


def test_strategy_b_embeds_expanded_query(mocker):
    embedder = mocker.Mock()
    embedder.embed_single.return_value = [0.0] * 384
    query_expander = mocker.Mock()
    query_expander.expand.return_value = "expanded query text"
    vector_store = mocker.Mock()
    vector_store.search.return_value = []

    retriever = RAGRetriever(vector_store, embedder, query_expander)
    retriever.retrieve_strategy_b("original query")

    embedder.embed_single.assert_called_once_with("expanded query text")


def test_results_have_correct_length():
    embedder = EmbeddingModel()
    vector_store = FAISSVectorStore(384)
    query_expander = QueryExpander()

    embeddings = embedder.get_embeddings(CORPUS)
    vector_store.add(CORPUS, embeddings)

    retriever = RAGRetriever(vector_store, embedder, query_expander)
    results_a = retriever.retrieve_strategy_a("test query", top_k=3)
    results_b = retriever.retrieve_strategy_b("test query", top_k=3)

    assert len(results_a) == 3
    assert len(results_b) == 3
