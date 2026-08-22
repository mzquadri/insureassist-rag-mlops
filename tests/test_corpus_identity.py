"""
Corpus identity: document hashes, chunk IDs, offsets, and idempotency.

Every ground-truth label names a chunk ID. If chunk IDs are not stable, the labels silently
stop pointing at the evidence they were written for and the benchmark measures nothing.
These tests exist to make that failure loud.
"""
import hashlib
from itertools import pairwise

import pytest

from src.corpus import (
    Document,
    chunk_corpus,
    chunk_document,
    config_hash,
    corpus_hash,
    load_corpus,
    load_manifest,
    make_chunk_id,
    validate_chunk_config,
    verify_document_hashes,
)

SIZE, OVERLAP = 600, 100


@pytest.fixture(scope="module")
def documents():
    return load_corpus()


@pytest.fixture(scope="module")
def chunks(documents):
    return chunk_corpus(documents, SIZE, OVERLAP)


class TestCorpusIntegrity:
    def test_manifest_lists_three_documents(self):
        assert len(load_manifest()["documents"]) == 3

    def test_documents_match_recorded_hashes(self, documents):
        assert verify_document_hashes(documents) == []

    def test_document_ids_are_the_expected_forms(self, documents):
        assert [d.document_id for d in documents] == [
            "nfip-sfip-dwelling",
            "nfip-sfip-general-property",
            "nfip-sfip-rcbap",
        ]

    def test_load_order_is_stable(self):
        assert [d.document_id for d in load_corpus()] == [d.document_id for d in load_corpus()]

    def test_a_modified_document_is_detected(self, documents):
        tampered = Document(
            document_id=documents[0].document_id,
            title=documents[0].title,
            form=documents[0].form,
            cfr_citation=documents[0].cfr_citation,
            text=documents[0].text + " tampered",
            sha256=documents[0].sha256,
        )
        assert verify_document_hashes([tampered]) == [documents[0].document_id]

    def test_corpus_hash_is_stable_and_order_sensitive(self, documents):
        assert corpus_hash(documents) == corpus_hash(load_corpus())
        assert corpus_hash(documents) != corpus_hash(list(reversed(documents)))

    def test_config_hash_changes_with_chunking(self):
        base = config_hash(embedding_model="m", size=600, overlap=100)
        assert base == config_hash(embedding_model="m", size=600, overlap=100)
        assert base != config_hash(embedding_model="m", size=500, overlap=100)
        assert base != config_hash(embedding_model="other", size=600, overlap=100)


class TestChunkIdentity:
    def test_chunk_count_is_the_documented_426(self, chunks):
        # docs/DATA.md and the reference run both state this figure.
        assert len(chunks) == 426

    def test_chunk_ids_are_stable_across_runs(self, documents):
        first = [c.chunk_id for c in chunk_corpus(documents, SIZE, OVERLAP)]
        second = [c.chunk_id for c in chunk_corpus(load_corpus(), SIZE, OVERLAP)]
        assert first == second

    def test_chunk_ids_are_unique(self, chunks):
        ids = [c.chunk_id for c in chunks]
        assert len(set(ids)) == len(ids)

    def test_point_ids_are_unique_and_deterministic(self, chunks):
        points = [c.point_id for c in chunks]
        assert len(set(points)) == len(points)
        assert points == [c.point_id for c in chunks]

    def test_no_uuid4_anywhere(self, chunks):
        """uuid4 would make every re-ingest produce new IDs. uuid5 must be reproducible."""
        again = chunk_corpus(load_corpus(), SIZE, OVERLAP)
        assert [c.point_id for c in again] == [c.point_id for c in chunks]

    def test_chunk_id_depends_on_document(self):
        """The forms share verbatim passages, so text alone must not determine the ID."""
        shared = "We will pay no more than $2,500 for any one loss"
        assert make_chunk_id("doc-a", 100, shared) != make_chunk_id("doc-b", 100, shared)

    def test_chunk_id_depends_on_offset(self):
        assert make_chunk_id("doc-a", 100, "text") != make_chunk_id("doc-a", 200, "text")

    def test_chunk_id_depends_on_text(self):
        assert make_chunk_id("doc-a", 100, "one") != make_chunk_id("doc-a", 100, "two")

    def test_chunk_id_is_namespaced_by_document(self, chunks):
        for chunk in chunks:
            assert chunk.chunk_id.startswith(f"{chunk.document_id}#")


class TestOffsets:
    def test_offsets_reproduce_chunk_text_exactly(self, documents, chunks):
        by_id = {d.document_id: d for d in documents}
        for chunk in chunks:
            assert by_id[chunk.document_id].text[chunk.start:chunk.end] == chunk.text

    def test_offsets_are_within_the_document(self, documents, chunks):
        by_id = {d.document_id: d for d in documents}
        for chunk in chunks:
            assert 0 <= chunk.start < chunk.end <= len(by_id[chunk.document_id].text)

    def test_chunk_indexes_are_contiguous_per_document(self, documents):
        for document in documents:
            indexes = [c.chunk_index for c in chunk_document(document, SIZE, OVERLAP)]
            assert indexes == list(range(len(indexes)))

    def test_chunks_advance_through_the_document(self, documents):
        for document in documents:
            starts = [c.start for c in chunk_document(document, SIZE, OVERLAP)]
            assert starts == sorted(starts)

    def test_whole_document_is_covered(self, documents):
        """No span of real text falls between two chunks."""
        for document in documents:
            document_chunks = chunk_document(document, SIZE, OVERLAP)
            for previous, following in pairwise(document_chunks):
                assert following.start <= previous.end


class TestChunkConfigValidation:
    def test_overlap_equal_to_size_is_rejected(self):
        with pytest.raises(ValueError, match="must be smaller than chunk size"):
            validate_chunk_config(600, 600)

    def test_overlap_greater_than_size_is_rejected(self):
        with pytest.raises(ValueError, match="must be smaller than chunk size"):
            validate_chunk_config(100, 500)

    def test_non_positive_size_is_rejected(self):
        with pytest.raises(ValueError, match="size must be positive"):
            validate_chunk_config(0, 0)

    def test_negative_overlap_is_rejected(self):
        with pytest.raises(ValueError, match="overlap cannot be negative"):
            validate_chunk_config(600, -1)

    def test_chunk_document_enforces_the_same_rules(self, documents):
        with pytest.raises(ValueError):
            chunk_document(documents[0], 100, 100)


class TestPayload:
    def test_payload_carries_full_traceability(self, documents, chunks):
        by_id = {d.document_id: d for d in documents}
        payload = chunks[0].payload(by_id[chunks[0].document_id])
        assert set(payload) == {
            "chunk_id", "document_id", "chunk_index", "start", "end",
            "text", "source", "form", "cfr_citation",
        }

    def test_payload_text_matches_the_source_file(self, documents, chunks):
        by_id = {d.document_id: d for d in documents}
        for chunk in chunks[:40]:
            document = by_id[chunk.document_id]
            payload = chunk.payload(document)
            assert document.text[payload["start"]:payload["end"]] == payload["text"]


class TestManifestMatchesCorpus:
    def test_recorded_hashes_are_real_sha256_of_the_files(self, documents):
        for document in documents:
            assert hashlib.sha256(document.text.encode("utf-8")).hexdigest() == document.sha256

    def test_recorded_counts_match(self, documents):
        manifest = {d["document_id"]: d for d in load_manifest()["documents"]}
        for document in documents:
            assert manifest[document.document_id]["characters"] == document.characters
            assert manifest[document.document_id]["words"] == len(document.text.split())
