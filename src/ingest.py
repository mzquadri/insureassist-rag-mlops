"""
Ingestion pipeline.

Reads the corpus, splits it into overlapping chunks that remember their offsets, embeds
each chunk, and stores them in Qdrant.

Run:
    python -m src.ingest                 # the real NFIP corpus (CORPUS=nfip)
    CORPUS=sample python -m src.ingest   # the synthetic sample documents

(Run it as a module, not as `python src/ingest.py` - the latter puts `src/` itself on the
path, so `from src.config import ...` cannot resolve.)

Ingestion is deterministic and idempotent. Chunk IDs are derived from the document, the
offset and the text, and the Qdrant point ID is a uuid5 of that chunk ID, so re-ingesting
an unchanged corpus overwrites exactly the same points instead of inserting duplicates
under fresh random IDs. The collection is only recreated when its vector size no longer
matches the embedder - it is no longer dropped on every run.
"""
import glob
import os

from src.config import cfg
from src.corpus import (
    Document,
    chunk_document,
    load_corpus,
    validate_chunk_config,
    verify_document_hashes,
)

#: Files that live in data/ but are documentation about the corpus, not part of it.
#: Without this, adding a provenance README to data/ silently indexes it as a policy
#: document and it starts turning up as a retrieval result.
NON_CORPUS_FILENAMES = {"README.md"}

BATCH_SIZE = 128

#: Points per scroll page when looking for records an earlier run left behind.
SCROLL_BATCH = 1000


def load_documents(data_dir: str = "data") -> list[dict]:
    """Load the synthetic sample .md / .txt files as {source, text}.

    Kept for the sample corpus and for tests. The real corpus is loaded through
    `src.corpus.load_corpus`, which reads a manifest rather than globbing a directory.
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
    """Fixed-size character chunking with overlap, returning text only.

    The offset-aware version used for the real corpus is `src.corpus.chunk_document`.
    """
    validate_chunk_config(size, overlap)

    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def sample_documents_as_corpus() -> list[Document]:
    """Wrap the synthetic sample files in the same Document type as the real corpus."""
    import hashlib

    documents = []
    for doc in load_documents():
        text = doc["text"]
        documents.append(
            Document(
                document_id=os.path.splitext(doc["source"])[0],
                title=doc["source"],
                form="sample",
                cfr_citation="",
                text=text,
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )
    return documents


def resolve_documents() -> list[Document]:
    """The corpus selected by configuration."""
    if cfg.CORPUS == "nfip":
        documents = load_corpus()
        mismatched = verify_document_hashes(documents)
        if mismatched:
            # A corpus file edited by hand silently invalidates every chunk ID, and with
            # them every ground-truth label. Refuse rather than index quietly-changed text.
            raise SystemExit(
                "Corpus files do not match data/corpus/manifest.json: "
                + ", ".join(mismatched)
                + "\nRun `python -m scripts.fetch_nfip_corpus` to rebuild."
            )
        return documents
    if cfg.CORPUS == "sample":
        return sample_documents_as_corpus()
    raise SystemExit(f"Unknown CORPUS {cfg.CORPUS!r}; expected 'nfip' or 'sample'")


def remove_stale_points(client, collection: str, current_ids: set) -> int:
    """Delete points the current corpus and chunking did not produce.

    Upserting is idempotent for an unchanged corpus, because a point ID is a uuid5 of a
    content-derived chunk ID, so re-running rewrites the same points. It is not sufficient
    on its own. Anything the previous run wrote and this one did not remains, and the
    collection was only ever recreated when the embedding dimension changed.

    Chunking is the case that matters. Ingesting the same three documents at 600/100 after
    a run at 800/120 left 314 points in place and added 426 more, and the collection then
    served 740 points drawn from two different chunkings at once. Nothing detected it: the
    closing line reported the total and asserted nothing.

    Scrolling IDs is affordable here because the corpus is a few hundred chunks. A corpus
    large enough for that to hurt would want a generation tag in the payload and a delete
    by filter instead.
    """
    from qdrant_client.models import PointIdsList

    stale: list = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        stale.extend(point.id for point in batch if point.id not in current_ids)
        if offset is None:
            break

    for start in range(0, len(stale), BATCH_SIZE):
        client.delete(
            collection_name=collection,
            points_selector=PointIdsList(points=stale[start:start + BATCH_SIZE]),
        )
    return len(stale)


def main():
    from qdrant_client.models import Distance, PointStruct, VectorParams

    from src.providers import get_embedder, get_vector_store

    print(f"Loading embedding model: {cfg.EMBEDDING_MODEL} ...")
    embedder = get_embedder()
    try:
        dim = embedder.get_embedding_dimension()
    except AttributeError:
        dim = embedder.get_sentence_embedding_dimension()

    client = get_vector_store()

    if client.collection_exists(cfg.QDRANT_COLLECTION):
        existing = client.get_collection(cfg.QDRANT_COLLECTION)
        current_dim = existing.config.params.vectors.size
        if current_dim != dim:
            print(f"Collection vector size {current_dim} != {dim}; recreating.")
            client.delete_collection(cfg.QDRANT_COLLECTION)
            client.create_collection(
                collection_name=cfg.QDRANT_COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
    else:
        client.create_collection(
            collection_name=cfg.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    print(f"Collection '{cfg.QDRANT_COLLECTION}' ready (dim={dim}).")

    documents = resolve_documents()
    points = []
    for document in documents:
        chunks = chunk_document(document, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
        # Passages are embedded WITHOUT the BGE query prefix; only queries carry it.
        vectors = embedder.encode([c.text for c in chunks], normalize_embeddings=True)
        for chunk, vector in zip(chunks, vectors):
            points.append(
                PointStruct(
                    id=chunk.point_id,
                    vector=vector.tolist(),
                    payload=chunk.payload(document),
                )
            )
        print(f"  {document.document_id}: {len(chunks)} chunks")

    for offset in range(0, len(points), BATCH_SIZE):
        client.upsert(
            collection_name=cfg.QDRANT_COLLECTION,
            points=points[offset:offset + BATCH_SIZE],
        )

    removed = remove_stale_points(client, cfg.QDRANT_COLLECTION, {p.id for p in points})
    if removed:
        print(f"  removed {removed} point(s) left by an earlier corpus or chunking")

    count = client.count(cfg.QDRANT_COLLECTION, exact=True).count
    print(f"Ingested {len(points)} chunks from {len(documents)} documents.")
    if count != len(points):
        # This line used to read "equal to chunk count if idempotent" and never checked.
        raise SystemExit(
            f"Collection holds {count} points but this run wrote {len(points)}. "
            "The collection does not mirror the corpus; refusing to report success."
        )
    print(f"Collection now holds {count} points, one per chunk.")


if __name__ == "__main__":
    main()
