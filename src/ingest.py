"""
Phase 1 — Ingestion pipeline.

Reads documents from data/, splits them into overlapping chunks, converts each chunk
into a vector (embedding) with a BGE model, and stores them in Qdrant so we can search
by meaning later.

Run:
    python src/ingest.py
"""
import glob
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from src.config import cfg


def load_documents(data_dir: str = "data") -> list[dict]:
    """Load every .md / .txt file in data/ as {source, text}."""
    docs = []
    for path in glob.glob(os.path.join(data_dir, "*.md")) + glob.glob(os.path.join(data_dir, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "text": f.read()})
    return docs


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Simple fixed-size character chunking with overlap (good enough to start)."""
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap  # step back by 'overlap' so context isn't cut mid-idea
    return chunks


def main():
    print(f"Loading embedding model: {cfg.EMBEDDING_MODEL} ...")
    embedder = SentenceTransformer(cfg.EMBEDDING_MODEL)
    # Vector size of this embedding model (handles both old/new method names).
    try:
        dim = embedder.get_embedding_dimension()
    except AttributeError:
        dim = embedder.get_sentence_embedding_dimension()

    client = QdrantClient(url=cfg.QDRANT_URL, check_compatibility=False)

    # Start the collection fresh: delete if it exists, then create it with the
    # right vector size + cosine similarity.
    if client.collection_exists(cfg.QDRANT_COLLECTION):
        client.delete_collection(cfg.QDRANT_COLLECTION)
    client.create_collection(
        collection_name=cfg.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"Collection '{cfg.QDRANT_COLLECTION}' ready (dim={dim}).")

    docs = load_documents()
    points = []
    for doc in docs:
        chunks = chunk_text(doc["text"], cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
        vectors = embedder.encode(chunks, normalize_embeddings=True)
        for chunk, vec in zip(chunks, vectors):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec.tolist(),
                    payload={"text": chunk, "source": doc["source"]},
                )
            )
        print(f"  {doc['source']}: {len(chunks)} chunks")

    client.upsert(collection_name=cfg.QDRANT_COLLECTION, points=points)
    print(f"Ingested {len(points)} chunks from {len(docs)} documents. Done.")


if __name__ == "__main__":
    main()
