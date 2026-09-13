"""A request must leave a trace, and the trace must not contain the question.

Both halves matter. Without timings, a slow Qdrant and a slow Ollama are indistinguishable
from the outside, which is the first thing anyone needs to know when the service is slow.
With the question in the log, an insurance assistant accumulates a file of what people
asked it about their homes and their claims, which is the last thing it should do.
"""

import logging

import pytest

from src import providers
from src.rag import answer

SECRET = "Is my basement flooding at 42 Example Street covered?"


@pytest.fixture
def logged(caplog):
    caplog.set_level(logging.INFO, logger="insureassist")
    return caplog


def test_an_answered_request_logs_both_durations(wired, logged):
    answer("What is the Increased Cost of Compliance limit?")
    lines = [r.getMessage() for r in logged.records if r.name == "insureassist"]
    assert len(lines) == 1, f"expected one log line, got {lines}"
    line = lines[0]
    assert "status=answered" in line
    assert "retrieval_ms=" in line
    assert "generation_ms=" in line
    assert "contexts=" in line


def test_an_abstention_logs_the_retrieval_time_and_no_generation_time(logged, fake_embedder):
    """Nothing was generated, so there is no generation duration to report."""
    from tests.conftest import FakeBM25, FakeVectorStore

    providers.set_embedder(fake_embedder)
    providers.set_vector_store(FakeVectorStore(hits=[]))
    providers.set_bm25_index(FakeBM25())
    providers.set_generator(lambda prompt: "should not be called")

    result = answer("A question with no evidence behind it")
    assert result["status"] == "insufficient_evidence"

    line = next(r.getMessage() for r in logged.records if r.name == "insureassist")
    assert "status=insufficient_evidence" in line
    assert "retrieval_ms=" in line
    assert "contexts=0" in line
    assert "generation_ms=" not in line


def test_the_question_text_never_reaches_the_log(wired, logged):
    answer(SECRET)
    combined = "\n".join(r.getMessage() for r in logged.records)
    assert SECRET not in combined
    assert "42 Example Street" not in combined
    assert "basement" not in combined.lower()


def test_the_log_reports_the_question_length_instead(wired, logged):
    """Enough to read a trace; not enough to reconstruct what was asked."""
    answer(SECRET)
    line = next(r.getMessage() for r in logged.records if r.name == "insureassist")
    assert f"question_chars={len(SECRET)}" in line


def test_the_answer_text_never_reaches_the_log(logged, fake_embedder):
    """The answer quotes the policy, but it is generated from a user's question."""
    from tests.conftest import FakeBM25, FakeVectorStore

    secret_answer = "Your basement claim at 42 Example Street is denied."
    providers.set_embedder(fake_embedder)
    providers.set_vector_store(FakeVectorStore())
    providers.set_bm25_index(FakeBM25())
    providers.set_generator(lambda prompt: secret_answer)

    answer("anything")
    combined = "\n".join(r.getMessage() for r in logged.records)
    assert secret_answer not in combined
    assert f"answer_chars={len(secret_answer)}" in combined


def test_durations_are_non_negative_numbers(wired, logged):
    import re

    answer("What is the Increased Cost of Compliance limit?")
    line = next(r.getMessage() for r in logged.records if r.name == "insureassist")
    for field in ("retrieval_ms", "generation_ms"):
        found = re.search(rf"{field}=([0-9.]+)", line)
        assert found, f"{field} missing from {line!r}"
        assert float(found.group(1)) >= 0


# -- the logger has to be enabled, or none of the above reaches a deployment --------------


def test_the_application_logger_is_configured_for_info():
    """uvicorn leaves the root logger at WARNING.

    Without src.api configuring its own logger, every line the tests above assert on is
    discarded in a deployed container. Verified that way round first: the timing line was
    absent from `docker logs` of a running container while the code that writes it was
    present and correct.
    """
    import src.api  # noqa: F401  - importing configures logging

    assert logging.getLogger("insureassist").isEnabledFor(logging.INFO)


def test_the_log_level_is_configurable():
    from src.config import Config

    assert hasattr(Config, "LOG_LEVEL")
    assert Config.LOG_LEVEL.upper() in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def test_configuring_twice_does_not_duplicate_handlers():
    """The API module can be imported more than once in a test session."""
    from src.api import _configure_logging

    lg = logging.getLogger("insureassist")
    before = len(lg.handlers)
    _configure_logging()
    _configure_logging()
    assert len(lg.handlers) == before
