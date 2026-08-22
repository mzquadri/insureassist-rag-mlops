"""
Ingestion determinism.

A scaffold, deliberately. Chunking is already deterministic and is asserted as such here.
Chunk *identity* is not: IDs are random UUIDs, so the same corpus ingested twice produces
different point IDs. That blocks labelled retrieval evaluation, because a relevance label
has to name a chunk that will still exist under the same name after the next ingest.

The gap is recorded as an executable expectation rather than a comment, so it fails the
moment content-derived IDs land and this file has to be finished.
"""
import uuid

import pytest

from src.ingest import chunk_text, load_documents


class TestChunkingIsDeterministic:
    def test_same_input_gives_same_chunks(self):
        text = "A clause. " * 200
        assert chunk_text(text, 600, 100) == chunk_text(text, 600, 100)

    def test_corpus_chunking_is_stable_across_runs(self):
        first = [chunk_text(d["text"], 600, 100) for d in load_documents("data")]
        second = [chunk_text(d["text"], 600, 100) for d in load_documents("data")]
        assert first == second

    def test_document_order_is_stable(self):
        assert [d["source"] for d in load_documents("data")] == \
               [d["source"] for d in load_documents("data")]


class TestChunkIdentity:
    """What ingestion must eventually guarantee, and does not yet."""

    def test_uuid4_ids_are_not_reproducible(self):
        """Documents today's behaviour: identical content, different identifiers."""
        assert str(uuid.uuid4()) != str(uuid.uuid4())

    @pytest.mark.xfail(
        reason="Chunk IDs are random UUIDs. Content-derived IDs are a later phase; "
               "until then chunks cannot be referenced by stable relevance labels.",
        strict=True,
    )
    def test_chunk_ids_are_derived_from_content(self):
        from src.ingest import chunk_id  # noqa: F401  - does not exist yet

        raise AssertionError("unreachable until chunk_id() is implemented")
