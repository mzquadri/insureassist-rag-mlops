"""
Ingestion pipeline.

Reads documents from data/, splits them into overlapping chunks, converts each chunk
into a vector (embedding) with a BGE model, and stores them in Qdrant so we can search
by meaning later.

Run:
    python -m src.ingest

(Run it as a module, not as `python src/ingest.py` - the latter puts `src/` itself on the
path, so `from src.config import ...` cannot resolve.)

Known limitation, tracked for a later phase: chunk IDs are random UUIDs, so re-ingesting
the same corpus produces different IDs every time. Ingestion is repeatable (the collection
is rebuilt from scratch) but not yet reproducible in the stronger sense that a given chunk
always carries the same identifier. Content-derived IDs are required before retrieval
results can be labelled and compared across runs.
"""
import glob
import os
import uuid

from src.config import cfg

#: Files that live in data/ but are documentation about the corpus, not part of it.
#: Without this, adding a provenance README to data/ silently indexes it as a policy
#: document and it starts turning up as a retrieval result.
NON_CORPUS_FILENAMES = {"README.md"}


def load_documents(data_dir: str = "data") -> list[dict]:
    """Load every corpus .md / .txt file in data/ as {source, text}.

    Sorted so ingestion visits documents in the same order every run.
    """
    docs = []
    for path in sorted(
        glob.glob(os.path.join(data_dir, "*.md")) + glob.glob(os.path.join(data_dir, "*.txt"))
    ):
        name = os.path.basename(path)
        if name in NON_CORPUS_FILENAMES:
            continue
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"source": name, "text": f.read()})
    return docs


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size character chunking with overlap.

    Each step advances by `size - overlap`. If that step is not positive the loop cannot
    make progress, so the arguments are rejected rather than allowed to hang: both values
    are configurable through the environment, and an overlap set to or above the chunk
    size would otherwise spin forever while appending the same text.
    """
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"chunk overlap cannot be negative, got {overlap}")
    if overlap >= size:
        raise ValueError(
            f"chunk overlap ({overlap}) must be smaller than chunk size ({size}); "
            "otherwise chunking cannot advance"
        )

    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap  # step back by 'overlap' so context isn't cut mid-idea
    return chunks


def main():
    # Imported here, not at module scope: sentence_transformers pulls in torch, and the
    # pure helpers above are used by tests that must stay fast and offline.
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from src.providers import get_embedder, get_vector_store

    print(f"Loading embedding model: {cfg.EMBEDDING_MODEL} ...")
    embedder = get_embedder()
    # Vector size of this embedding model (handles both old/new method names).
    try:
        dim = embedder.get_embedding_dimension()
    except AttributeError:
        dim = embedder.get_sentence_embedding_dimension()

    client = get_vector_store()

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
