"""
Corpus loading and deterministic chunking.

Every chunk is traceable: document -> exact character offsets -> exact source text ->
stable chunk ID. That chain is what makes labelled retrieval evaluation possible, because a
relevance label has to name a chunk that still exists, unchanged, after the next ingest.

Ingestion used to assign `uuid.uuid4()` to each point, so the same corpus ingested twice
produced different identifiers and no label could survive. Chunk IDs are now derived from
the document, the offset, and the text itself.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path("data/corpus")
MANIFEST_PATH = CORPUS_DIR / "manifest.json"

#: Fixed namespace so chunk UUIDs are reproducible across machines and runs. Qdrant point
#: IDs must be a UUID or an unsigned integer, so the readable chunk ID is mapped through
#: uuid5 rather than replaced by a random one. Re-ingesting overwrites the same point,
#: which is what makes ingestion idempotent instead of merely repeatable.
POINT_NAMESPACE = uuid.UUID("6f6e0d3a-4a5b-5c6d-8e9f-0a1b2c3d4e5f")


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    form: str
    cfr_citation: str
    text: str
    sha256: str

    @property
    def characters(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    start: int
    end: int
    text: str

    @property
    def point_id(self) -> str:
        """Deterministic UUID for this chunk, used as the Qdrant point ID."""
        return str(uuid.uuid5(POINT_NAMESPACE, self.chunk_id))

    def payload(self, document: Document) -> dict:
        """What gets stored alongside the vector.

        Carries enough to rebuild the citation and to verify the chunk against the source
        file without consulting anything else.
        """
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            # `source` is kept for compatibility with the existing prompt builder, which
            # labels context blocks with it.
            "source": document.title,
            "form": document.form,
            "cfr_citation": document.cfr_citation,
        }


def validate_chunk_config(size: int, overlap: int) -> None:
    """Reject configurations that cannot chunk.

    The cursor advances by `size - overlap`. A non-positive step never terminates, and both
    values are set from the environment, so this is reachable by configuration alone.
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


def make_chunk_id(document_id: str, start: int, text: str) -> str:
    """A stable identifier derived from where the text is and what it says.

    The document ID and the offset are both part of the hash on purpose. The three NFIP
    forms contain passages that are identical word for word, so hashing the text alone
    would collide across documents and silently merge distinct chunks.
    """
    digest = hashlib.sha256(f"{document_id}|{start}|{text}".encode()).hexdigest()
    return f"{document_id}#{digest[:12]}"


def chunk_document(document: Document, size: int, overlap: int) -> list[Chunk]:
    """Split a document into overlapping chunks that remember where they came from.

    Offsets index the document's own text, so `document.text[chunk.start:chunk.end]` is
    exactly `chunk.text`. Leading and trailing whitespace is trimmed from each window and
    the offsets are moved to match, rather than trimmed text being paired with untrimmed
    offsets.
    """
    validate_chunk_config(size, overlap)

    chunks: list[Chunk] = []
    cursor = 0
    index = 0
    text = document.text

    while cursor < len(text):
        window_end = min(cursor + size, len(text))
        window = text[cursor:window_end]
        stripped = window.strip()

        if stripped:
            start = cursor + (len(window) - len(window.lstrip()))
            end = start + len(stripped)
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(document.document_id, start, stripped),
                    document_id=document.document_id,
                    chunk_index=index,
                    start=start,
                    end=end,
                    text=stripped,
                )
            )
            index += 1

        if window_end == len(text):
            break
        cursor = window_end - overlap

    return chunks


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[Document]:
    """Load every document named in the manifest, in manifest order.

    Reading from the manifest rather than globbing the directory keeps the corpus a closed,
    recorded set: a stray file cannot wander into the index, and the order is fixed.
    """
    manifest = load_manifest(corpus_dir / "manifest.json")
    documents = []
    for entry in manifest["documents"]:
        text = (corpus_dir / entry["filename"]).read_text(encoding="utf-8")
        documents.append(
            Document(
                document_id=entry["document_id"],
                title=entry["title"],
                form=entry["form"],
                cfr_citation=entry["cfr_citation"],
                text=text,
                sha256=entry["sha256"],
            )
        )
    return documents


def chunk_corpus(documents: list[Document], size: int, overlap: int) -> list[Chunk]:
    """Chunk every document, preserving document order and chunk order within a document."""
    return [chunk for document in documents for chunk in chunk_document(document, size, overlap)]


def verify_document_hashes(documents: list[Document]) -> list[str]:
    """Return the IDs of any documents whose text no longer matches the recorded hash."""
    mismatched = []
    for document in documents:
        actual = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
        if actual != document.sha256:
            mismatched.append(document.document_id)
    return mismatched


def corpus_hash(documents: list[Document]) -> str:
    """One hash covering the whole corpus, for stamping a reference run."""
    joined = "|".join(f"{d.document_id}:{d.sha256}" for d in documents)
    return hashlib.sha256(joined.encode()).hexdigest()


def config_hash(*, embedding_model: str, size: int, overlap: int) -> str:
    """Hash of the settings that determine what the index contains."""
    return hashlib.sha256(f"{embedding_model}|{size}|{overlap}".encode()).hexdigest()
