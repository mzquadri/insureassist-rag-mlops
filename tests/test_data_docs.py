"""`data/README.md` describes the sample corpus. The description must stay true.

It had drifted twice. It documented "chunk size of 600 characters with 100 characters of
overlap", producing "7 chunks in total" at "TOP_K=4", while the configured defaults were
800/120 and top-5, which produce 6. And it ended "until it lands, no retrieval metric is
published anywhere in this repository", a sentence that stopped being true when the NFIP
corpus landed and the README began publishing hit rate and MRR measured on it.

Neither was a typo. Both were statements that were correct when written and were never
revisited, which is what documentation does unless something checks it.
"""

import re
from pathlib import Path

from src.config import cfg
from src.ingest import chunk_text, load_documents

ROOT = Path(__file__).resolve().parents[1]
DATA_README = ROOT / "data" / "README.md"


def sample_chunk_count() -> int:
    """Chunks the two synthetic documents produce at the configured settings."""
    return sum(
        len(chunk_text(doc["text"], cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP))
        for doc in load_documents(str(ROOT / "data"))
    )


def test_the_documented_chunk_settings_are_the_configured_ones():
    text = DATA_README.read_text(encoding="utf-8")
    assert f"chunk size of {cfg.CHUNK_SIZE} characters" in text
    assert f"{cfg.CHUNK_OVERLAP} characters of overlap" in text
    assert f"`TOP_K={cfg.TOP_K}`" in text


def test_the_documented_chunk_count_is_what_the_corpus_produces():
    text = DATA_README.read_text(encoding="utf-8")
    stated = re.search(r"\*\*(\d+) chunks in total\*\*", text)
    assert stated, "data/README.md no longer states a chunk total"
    assert int(stated.group(1)) == sample_chunk_count()


def test_the_sample_corpus_is_still_smaller_than_the_retrieval_window():
    """The point the section exists to make: top-k covers most of the index.

    If a larger sample corpus ever makes this false, the warning is obsolete and the
    section needs rewriting rather than quietly becoming wrong.
    """
    assert cfg.TOP_K >= sample_chunk_count() - 1, (
        "the sample corpus has outgrown the retrieval window; the recall warning in "
        "data/README.md needs rewriting"
    )


def test_the_synthetic_files_are_declared_as_synthetic():
    text = DATA_README.read_text(encoding="utf-8")
    assert "synthetic and self-authored" in text
    for name in ("auto_insurance_policy.md", "home_insurance_policy.md", "qa_testset.jsonl"):
        assert name in text, f"{name} is not accounted for in data/README.md"


def test_it_no_longer_claims_that_no_retrieval_metric_is_published():
    """The NFIP corpus landed and the README publishes metrics measured on it."""
    text = DATA_README.read_text(encoding="utf-8")
    assert "no retrieval metric is published anywhere in this repository" not in text


def test_the_sample_documents_carry_no_contact_or_identity_data():
    """They are invented, and must stay free of anything that looks personal."""
    patterns = {
        "email": r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "phone": r"\b\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b",
    }
    for name in ("auto_insurance_policy.md", "home_insurance_policy.md", "qa_testset.jsonl"):
        body = (ROOT / "data" / name).read_text(encoding="utf-8")
        for label, pattern in patterns.items():
            assert not re.search(pattern, body), f"{name} contains something shaped like a {label}"
