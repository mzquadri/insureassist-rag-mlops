"""
Lazy, overridable access to the heavy runtime dependencies.

Before this module existed, `src/rag.py` built a SentenceTransformer and a QdrantClient at
import time. That had three consequences worth spelling out, because they are the reason
this seam exists:

  * `import src.api` downloaded an embedding model. A machine with no network could not
    import the application at all, let alone test it.
  * A model or database failure became an ImportError during startup rather than a handled
    503 at request time, so the process crash-looped instead of reporting what was wrong.
  * No test could exercise the API without the real model and a live Qdrant.

Everything heavy is therefore constructed on first use and cached, and every construction
site can be replaced by a test. This is a seam, not a framework: three module-level slots
and three setters. Nothing here should grow into a container or a registry.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from src.config import cfg
from src.errors import GenerationUnavailable, RetrievalUnavailable


class Embedder(Protocol):
    """The slice of SentenceTransformer this project actually uses."""

    def encode(self, sentences, normalize_embeddings: bool = False):  # pragma: no cover
        ...


class VectorStore(Protocol):
    """The slice of QdrantClient the query path actually uses."""

    def query_points(self, *, collection_name, query, limit, with_payload):  # pragma: no cover
        ...


#: A generator turns a finished prompt into text. Deliberately a plain callable so a test
#: can pass a lambda and a backend can be swapped without implementing an interface.
Generator = Callable[[str], str]


_embedder: Embedder | None = None
_bm25 = None
_vector_store: VectorStore | None = None
_generator: Generator | None = None


def get_embedder() -> Embedder:
    """The sentence embedder, loaded on first use."""
    global _embedder
    if _embedder is None:
        try:
            # Imported here rather than at module scope: importing sentence_transformers
            # pulls in torch, which is slow and unnecessary for anything that only needs
            # the pure helpers in this package.
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer(cfg.EMBEDDING_MODEL)
        except Exception as exc:  # boundary: any load failure is a 503
            raise RetrievalUnavailable(
                f"could not load embedding model {cfg.EMBEDDING_MODEL!r}"
            ) from exc
    return _embedder


def get_vector_store() -> VectorStore:
    """The Qdrant client, constructed on first use."""
    global _vector_store
    if _vector_store is None:
        try:
            from qdrant_client import QdrantClient

            _vector_store = QdrantClient(url=cfg.QDRANT_URL, check_compatibility=False)
        except Exception as exc:  # boundary: this is an external service
            raise RetrievalUnavailable(
                f"could not connect to Qdrant at {cfg.QDRANT_URL}"
            ) from exc
    return _vector_store


def _ollama_generator(prompt: str) -> str:
    """Generate with a local Ollama model."""
    import requests

    try:
        resp = requests.post(
            f"{cfg.OLLAMA_URL}/api/generate",
            json={"model": cfg.OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except Exception as exc:  # boundary: this is an external service
        raise GenerationUnavailable(
            f"generator {cfg.OLLAMA_MODEL!r} at {cfg.OLLAMA_URL} did not answer"
        ) from exc


def _hf_generator(prompt: str) -> str:
    """
    Generate with the Hugging Face path (base model, plus a LoRA adapter if one is present).

    No adapter ships with this repository, so unless you have trained and copied one in,
    this path runs the *base* model. See `finetune/README.md`.
    """
    try:
        from src.hf_generator import generate_hf

        return generate_hf(prompt)
    except GenerationUnavailable:
        raise
    except Exception as exc:  # boundary: this is an external service
        raise GenerationUnavailable(
            f"Hugging Face backend failed for base model {cfg.HF_BASE_MODEL!r}"
        ) from exc


def get_generator() -> Generator:
    """The configured generator, resolved on first use."""
    global _generator
    if _generator is None:
        backends: dict[str, Generator] = {
            "ollama": _ollama_generator,
            "hf": _hf_generator,
        }
        try:
            _generator = backends[cfg.LLM_BACKEND]
        except KeyError as exc:
            # A typo in configuration is a deployment error, not a transient outage.
            raise ValueError(
                f"Unknown LLM_BACKEND {cfg.LLM_BACKEND!r}; expected one of {sorted(backends)}"
            ) from exc
    return _generator


def get_bm25_index():
    """The lexical index, built on first use from the committed corpus.

    Built in-process rather than stored in Qdrant: the corpus is 3 documents and a few
    hundred chunks, so the index costs milliseconds to construct and needs no extra
    service. It is cached like the other dependencies and replaceable by a test.
    """
    global _bm25
    if _bm25 is None:
        try:
            from src.corpus import chunk_corpus, load_corpus
            from src.retrieval import BM25Index

            documents = load_corpus()
            chunks = chunk_corpus(documents, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
            _bm25 = BM25Index(chunks)
        except Exception as exc:  # boundary: corpus missing or unreadable
            raise RetrievalUnavailable("could not build the lexical index") from exc
    return _bm25


def set_bm25_index(index) -> None:
    """Install a lexical index (tests, or a prebuilt one)."""
    global _bm25
    _bm25 = index


def set_embedder(embedder: Embedder | None) -> None:
    """Install an embedder (tests, or a preloaded model)."""
    global _embedder
    _embedder = embedder


def set_vector_store(store: VectorStore | None) -> None:
    """Install a vector store (tests, or an already-configured client)."""
    global _vector_store
    _vector_store = store


def set_generator(generator: Generator | None) -> None:
    """Install a generator (tests, or a backend chosen at runtime)."""
    global _generator
    _generator = generator


def reset() -> None:
    """Drop every cached dependency. Tests call this so state cannot leak between them."""
    set_embedder(None)
    set_vector_store(None)
    set_generator(None)
    set_bm25_index(None)
