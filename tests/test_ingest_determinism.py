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
    """The guarantee that makes labelled evaluation possible.

    This file previously carried a strict xfail here, marking content-derived IDs as an open
    gap. They are implemented, so the expectation is now asserted directly.
    """

    def test_uuid4_would_not_be_reproducible(self):
        """Why uuid4 was unusable: identical content, different identifiers every call."""
        assert str(uuid.uuid4()) != str(uuid.uuid4())

    def test_chunk_ids_are_derived_from_content(self):
        from src.corpus import make_chunk_id

        assert make_chunk_id("doc", 0, "text") == make_chunk_id("doc", 0, "text")
        assert make_chunk_id("doc", 0, "text") != make_chunk_id("doc", 1, "text")

    def test_point_ids_survive_reingestion(self):
        """A relevance label names a chunk; the chunk must still exist under that name."""
        from src.config import cfg
        from src.corpus import chunk_corpus, load_corpus

        first = {c.chunk_id: c.point_id for c in
                 chunk_corpus(load_corpus(), cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)}
        second = {c.chunk_id: c.point_id for c in
                  chunk_corpus(load_corpus(), cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)}
        assert first == second and len(first) > 300
