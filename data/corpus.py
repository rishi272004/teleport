CORPUS = [
    "Horizontal scaling distributes incoming traffic across multiple server instances using a load balancer. During peak load, the system automatically provisions additional compute nodes through an auto-scaling group triggered by CPU utilization thresholds exceeding 70 percent.",

    "The vector database uses FAISS with an IVF (Inverted File Index) structure, enabling approximate nearest neighbor search across millions of embeddings in milliseconds. Index partitioning into Voronoi cells reduces search space significantly.",

    "Query expansion is a technique where the original user query is rewritten or augmented with semantically related terms before embedding. This improves retrieval recall by bridging the vocabulary gap between user intent and document language.",

    "Cosine similarity measures the angle between two embedding vectors, making it invariant to magnitude. It is preferred over Euclidean distance for semantic search because it captures directional alignment in high-dimensional space rather than absolute distance.",

    "Rate limiting and circuit breakers are essential patterns for handling traffic spikes. When downstream services become overwhelmed, the circuit breaker trips and returns cached or degraded responses, preventing cascading failures across the system.",

    "Retrieval-Augmented Generation (RAG) combines a retrieval component with a generative language model. The retrieval step fetches relevant document chunks from a vector store, and the generator synthesizes a grounded response conditioned on those chunks.",

    "Embedding models convert raw text into dense floating-point vectors that encode semantic meaning. Models like sentence-transformers/all-MiniLM-L6-v2 produce 384-dimensional vectors that cluster semantically similar sentences close together in embedding space.",

    "In production environments, Vertex AI Vector Search (formerly Matching Engine) supports billion-scale approximate nearest neighbor (ANN) search with sub-millisecond latency. It integrates natively with Vertex AI embeddings and supports real-time and batch update modes.",

    "Chunking strategy directly impacts retrieval quality. Fixed-size chunking splits documents into equal token windows, while semantic chunking uses sentence boundaries and topic shifts. Overlapping chunks ensure context is not lost at chunk boundaries.",

    "Observability in ML pipelines includes tracking embedding drift, retrieval latency, and answer faithfulness scores. Tools like LangSmith and MLflow log intermediate retrieval steps, enabling debugging of retrieval failures in production RAG systems.",
]
