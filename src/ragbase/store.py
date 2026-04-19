from dataclasses import dataclass
import chromadb
from ragbase.ingestion import Chunk
from ragbase.embedder import embed_texts, embed_query

CHROMA_DIR = "chroma_db"


@dataclass
class SearchResult:
    text: str
    page: int
    score: float


def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_DIR)


def _collection_name(doc_id: str) -> str:
    safe = "".join(c if c.isalnum() else "-" for c in doc_id)
    return safe[:63] or "doc"


def index_chunks(doc_id: str, chunks: list[Chunk]) -> None:
    client = _get_client()
    name = _collection_name(doc_id)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    collection.add(
        ids=[f"{doc_id}_{c.chunk_index}" for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"page": c.page, "chunk_index": c.chunk_index} for c in chunks],
    )


def query_store(doc_id: str, query: str, top_k: int = 5) -> list[SearchResult]:
    client = _get_client()
    name = _collection_name(doc_id)
    collection = client.get_collection(name)
    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    search_results = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        search_results.append(SearchResult(text=doc, page=meta["page"], score=1 - dist))
    return search_results
