"""
Shared fixtures.

Every test in this suite is offline. Nothing here downloads a model, opens a socket, or
expects Ollama or Qdrant to be running - which is only possible because `src.providers`
builds those things lazily and lets a test replace them.
"""
import sys
import types

import pytest

from src import providers


@pytest.fixture(autouse=True)
def _reset_providers():
    """Clear cached dependencies around every test so state cannot leak between them."""
    providers.reset()
    yield
    providers.reset()


class FakeVector(list):
    """A vector that answers `.tolist()`, which is all the retrieval path asks of it."""

    def tolist(self):
        return list(self)


class FakeEmbedder:
    """Stands in for SentenceTransformer. Records what it was asked to embed."""

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.calls: list = []

    def encode(self, sentences, normalize_embeddings: bool = False):
        self.calls.append(sentences)
        if isinstance(sentences, str):
            return FakeVector([0.1] * self.dim)
        return [FakeVector([0.1] * self.dim) for _ in sentences]


def _hit(text: str, source: str, score: float):
    return types.SimpleNamespace(payload={"text": text, "source": source}, score=score)


class FakeVectorStore:
    """Stands in for QdrantClient's query path."""

    def __init__(self, hits=None, error: Exception | None = None):
        self._hits = hits if hits is not None else [
            _hit("Burst pipe damage is covered.", "home_insurance_policy.md", 0.8712),
            _hit("A higher excess of EUR 500 applies.", "home_insurance_policy.md", 0.7031),
        ]
        self._error = error
        self.queries: list = []

    def query_points(self, *, collection_name, query, limit, with_payload):
        if self._error:
            raise self._error
        self.queries.append({"collection": collection_name, "limit": limit})
        return types.SimpleNamespace(points=self._hits[:limit])


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_store():
    return FakeVectorStore()


@pytest.fixture
def make_store():
    """Build a store with custom hits, or one that fails, without importing this module."""
    def _make(hits=None, error: Exception | None = None):
        return FakeVectorStore(hits=hits, error=error)

    return _make


@pytest.fixture
def wired(fake_embedder, fake_store):
    """A fully stubbed pipeline: embedder, store, and a generator that echoes."""
    providers.set_embedder(fake_embedder)
    providers.set_vector_store(fake_store)
    providers.set_generator(lambda prompt: "Yes, a burst pipe is covered.")
    return types.SimpleNamespace(embedder=fake_embedder, store=fake_store)


@pytest.fixture
def heavy_modules_unloaded():
    """Names that must not appear in sys.modules merely because the app was imported."""
    return [name for name in ("sentence_transformers", "torch", "qdrant_client")
            if name not in sys.modules]
