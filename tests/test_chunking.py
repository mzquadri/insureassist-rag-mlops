"""Chunking: normal behaviour, boundaries, and the configurations that must be refused."""
import pytest

from src.ingest import chunk_text, load_documents


class TestNormalCases:
    def test_short_text_is_a_single_chunk(self):
        assert chunk_text("hello world", size=600, overlap=100) == ["hello world"]

    def test_text_longer_than_size_is_split(self):
        chunks = chunk_text("a" * 1000, size=600, overlap=100)
        assert len(chunks) == 2
        assert len(chunks[0]) == 600

    def test_chunks_overlap_by_the_requested_amount(self):
        text = "".join(str(i % 10) for i in range(1000))
        chunks = chunk_text(text, size=600, overlap=100)
        # The second chunk restarts 100 characters before the first one ended.
        assert chunks[1][:100] == chunks[0][-100:]

    def test_advances_by_size_minus_overlap(self):
        text = "x" * 2000
        chunks = chunk_text(text, size=500, overlap=100)
        # step = 400, so starts land on 0, 400, 800, 1200, 1600 -> 5 chunks
        assert len(chunks) == 5

    def test_is_deterministic(self):
        text = "clause " * 300
        assert chunk_text(text, 600, 100) == chunk_text(text, 600, 100)

    def test_zero_overlap_is_allowed(self):
        chunks = chunk_text("a" * 100, size=10, overlap=0)
        assert len(chunks) == 10
        assert "".join(chunks) == "a" * 100


class TestEdgeCases:
    def test_empty_text_yields_no_chunks(self):
        assert chunk_text("", size=600, overlap=100) == []

    def test_whitespace_only_text_yields_no_chunks(self):
        assert chunk_text("   \n\t  ", size=600, overlap=100) == []

    def test_chunks_are_stripped(self):
        assert all(c == c.strip() for c in chunk_text("  padded text  " * 80, 100, 10))

    def test_text_exactly_one_chunk_long_emits_a_duplicate_tail(self):
        """Pins a real wart: text of exactly `size` produces a second, redundant chunk.

        After emitting characters 0-600 the cursor steps back to 500, which is still inside
        the text, so a final 100-character chunk is produced that is wholly contained in the
        first one. It inflates the index and can compete with its own parent at retrieval
        time. Recorded rather than silently changed - altering chunk boundaries would
        invalidate the recorded evaluation report, so it belongs with the chunking rework.
        """
        chunks = chunk_text("a" * 600, size=600, overlap=100)
        assert len(chunks) == 2
        assert chunks[1] == "a" * 100
        assert chunks[1] in chunks[0]

    def test_size_larger_than_text(self):
        assert chunk_text("short", size=10_000, overlap=100) == ["short"]


class TestRejectedConfigurations:
    """Both values come from the environment, so a bad pair must fail loudly.

    Before this guard, `overlap >= size` made the loop step by zero or backwards: it never
    terminated and appended the same text until the process died.
    """

    def test_overlap_equal_to_size_is_rejected(self):
        with pytest.raises(ValueError, match="must be smaller than chunk size"):
            chunk_text("some text", size=600, overlap=600)

    def test_overlap_greater_than_size_is_rejected(self):
        with pytest.raises(ValueError, match="must be smaller than chunk size"):
            chunk_text("some text", size=100, overlap=250)

    def test_zero_size_is_rejected(self):
        with pytest.raises(ValueError, match="size must be positive"):
            chunk_text("some text", size=0, overlap=0)

    def test_negative_size_is_rejected(self):
        with pytest.raises(ValueError, match="size must be positive"):
            chunk_text("some text", size=-10, overlap=0)

    def test_negative_overlap_is_rejected(self):
        with pytest.raises(ValueError, match="overlap cannot be negative"):
            chunk_text("some text", size=100, overlap=-1)

    def test_rejection_happens_before_any_work(self):
        # A guard that only triggered inside the loop would still hang on empty input.
        with pytest.raises(ValueError):
            chunk_text("", size=100, overlap=100)


class TestCorpusShape:
    """Pins the documented size of the sample corpus.

    `data/README.md` and the README both state that the sample corpus is 7 chunks, and use
    that number to explain why retrieval cannot be measured on it. If the corpus or the
    chunking changes, that reasoning needs revisiting, so the claim is tested rather than
    left as prose.
    """

    def test_sample_corpus_produces_seven_chunks(self):
        docs = load_documents("data")
        assert [d["source"] for d in docs] == [
            "auto_insurance_policy.md",
            "home_insurance_policy.md",
        ]
        total = sum(len(chunk_text(d["text"], 600, 100)) for d in docs)
        assert total == 7

    def test_load_documents_is_ordered(self):
        # Stable order keeps ingestion comparable between runs.
        sources = [d["source"] for d in load_documents("data")]
        assert sources == sorted(sources)
