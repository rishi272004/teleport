# RAG Vector Search - Senior GenAI Assessment

## Setup
```bash
pip install -r requirements.txt
```

## Run Benchmark
```bash
python run_benchmark.py
```
Output: prints table to stdout + writes `retrieval_benchmark.md`

## Run Tests
```bash
pytest tests/ -v
```

## Architecture
- **Embedder**: sentence-transformers (mocks Vertex AI textembedding-gecko)
- **Vector Store**: FAISS IndexFlatIP + L2 normalization (cosine similarity)
- **Query Expander**: deterministic keyword expansion (mocks Vertex AI GenerativeModel)
- **Strategies**: A = raw embed -> search | B = expand -> embed -> search

## Design Decisions
- Cosine over Euclidean: magnitude-invariant, superior for high-dim semantic search
- FAISS flat index: exact search, appropriate for small corpus (<10k docs)
- Mocking at class level: all GCP SDK calls are mocked behind class boundaries,
  making tests fast, offline, and deterministic
