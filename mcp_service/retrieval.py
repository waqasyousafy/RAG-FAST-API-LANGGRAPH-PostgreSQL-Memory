"""
Retrieval Layer — attaches to an already-ingested vector store, runs MMR
search and cross-encoder reranking. Owned exclusively by the MCP server.

Ingestion is handled separately by ingest.py — run that manually whenever
documents change. This file never writes to the vector store.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from sentence_transformers import CrossEncoder
from langsmith import traceable
from config import DB_CONNECTION, COLLECTION_NAME

_db_vector = None
_reranker = None
_embedding_function = None


def get_embedding_function():
    global _embedding_function
    if _embedding_function is None:
        print("Loading embedding model...")
        _embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embedding_function


def get_reranker():
    global _reranker
    if _reranker is None:
        print("Loading reranker model...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def get_vector_store():
    """
    Attach to the EXISTING collection — no re-embedding, no re-insertion.
    This is what makes server startup fast: it's a connection, not a build.
    """
    global _db_vector
    if _db_vector is None:
        print(f"Attaching to existing vector store collection '{COLLECTION_NAME}'...")
        _db_vector = PGVector(
            embeddings=get_embedding_function(),
            collection_name=COLLECTION_NAME,
            connection=DB_CONNECTION,
            use_jsonb=True,
        )
        print("Vector store attached.")
    return _db_vector


@traceable(name="vector_retrieval", run_type="retriever")
def retrieve_candidates(query: str, k: int = 3, fetch_k: int = 5):
    vector_store = get_vector_store()
    results = vector_store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)
    print(f"[retrieve_candidates] query={query!r} -> {len(results)} raw hits")
    return results


@traceable(name="cross_encoder_rerank", run_type="chain")
def rerank_documents(query: str, documents, top_n: int = 3):
    if not documents:
        return []
    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
    print(f"[rerank_documents] scores={[float(s) for s, _ in scored_docs]}")
    return [doc for score, doc in scored_docs[:top_n]]


def search_documents_raw(query: str) -> str:
    """Retrieval + rerank + format — the actual work behind the MCP tool."""
    results = retrieve_candidates(query, k=3, fetch_k=5)
    if not results:
        return "No relevant documents found."

    top_results = rerank_documents(query, results, top_n=3)

    MAX_CHARS_PER_CHUNK = 500
    formatted = [
        f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content[:MAX_CHARS_PER_CHUNK]}"
        for doc in top_results
    ]
    return "\n\n".join(formatted)