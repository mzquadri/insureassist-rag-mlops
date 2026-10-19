"""The LangGraph flow: routing, both policies, and the dependency injection.

No network, no Qdrant, no LLM. Retrieval and generation are both injected, which
is the property that makes the decision testable on its own - and the same
property `eval/graph_abstention.py` relies on to measure 50 questions without a
generator.
"""

from __future__ import annotations

import pytest

langgraph = pytest.importorskip(
    "langgraph", reason="langgraph is optional; pip install -r requirements-graph.txt"
)

from src import graph as g


def context(chunk_id: str = "c1", dense_score: float | None = 0.80, text: str = "body") -> dict:
    """One retrieved context, with only the fields the graph reads."""
    return {
        "chunk_id": chunk_id,
        "document_id": "nfip-sfip-dwelling",
        "source": "nfip-sfip-dwelling",
        "form": "Dwelling Form",
        "cfr_citation": "44 CFR Part 61",
        "start": 0,
        "end": len(text),
        "text": text,
        "score": 0.5,
        "dense_score": dense_score,
        "lexical_score": 1.0,
    }


def run(contexts, policy=g.POLICY_NO_EVIDENCE, threshold=None, generated="an answer"):
    """Invoke the graph with retrieval and generation both stubbed."""
    calls = {"generate": 0}

    def fake_generate(prompt: str) -> str:
        calls["generate"] += 1
        return generated

    compiled = g.build_graph(
        generate=fake_generate,
        retrieve=lambda question: contexts,
    )
    state = compiled.invoke({"question": "does the policy cover flood?", "policy": policy,
                             "threshold": threshold})
    return state, calls


# ------------------------------------------------------------------ routing

def test_no_contexts_abstains_under_every_policy():
    for policy in g.POLICIES:
        state, calls = run([], policy=policy)
        assert state["decision"] == "abstain", policy
        assert state["status"] == "insufficient_evidence", policy
        assert state["citations"] == []
        # The generator must not be reached. Abstaining after paying for a
        # generation would defeat the point of abstaining.
        assert calls["generate"] == 0, policy


def test_contexts_answer_under_the_shipped_policy():
    state, calls = run([context()], policy=g.POLICY_NO_EVIDENCE)

    assert state["decision"] == "answer"
    assert state["status"] == "answered"
    assert state["answer"] == "an answer"
    assert calls["generate"] == 1
    assert len(state["citations"]) == 1


def test_the_shipped_policy_ignores_the_score_entirely():
    # A very low dense score still answers, because POLICY_NO_EVIDENCE is not a
    # threshold rule. This is the behaviour eval/abstention_threshold.json
    # argued for, so it is pinned rather than left implicit.
    state, _ = run([context(dense_score=0.01)], policy=g.POLICY_NO_EVIDENCE)
    assert state["decision"] == "answer"


# ------------------------------------------------- the rejected threshold

def test_threshold_policy_abstains_below_and_answers_above():
    low, _ = run([context(dense_score=0.70)], policy=g.POLICY_THRESHOLD, threshold=0.75)
    high, _ = run([context(dense_score=0.80)], policy=g.POLICY_THRESHOLD, threshold=0.75)

    assert low["decision"] == "abstain"
    assert "below threshold" in low["reason"]
    assert high["decision"] == "answer"


def test_threshold_boundary_answers():
    # At exactly the threshold the comparison is `top < threshold`, so the
    # boundary answers. Stated as a test because either choice is defensible and
    # the eval's counts depend on which one it is.
    state, _ = run([context(dense_score=0.75)], policy=g.POLICY_THRESHOLD, threshold=0.75)
    assert state["decision"] == "answer"


def test_threshold_reads_the_highest_dense_score_not_the_first():
    # The fused order is by RRF score, not by dense score, so the first context
    # is not necessarily the highest-scoring one. The signal the study used is
    # "top-1 dense cosine over the retrieved contexts".
    contexts = [context("c1", dense_score=0.60), context("c2", dense_score=0.90)]
    state, _ = run(contexts, policy=g.POLICY_THRESHOLD, threshold=0.75)
    assert state["decision"] == "answer"


def test_threshold_defaults_to_the_dev_selected_value():
    state, _ = run([context(dense_score=0.70)], policy=g.POLICY_THRESHOLD, threshold=None)
    assert state["decision"] == "abstain"
    assert f"{g.DEV_SELECTED_THRESHOLD:.4f}" in state["reason"]


def test_missing_dense_scores_answer_rather_than_abstain():
    # A chunk that reached the fused list from BM25 only has no dense score.
    # Abstaining then would be abstaining because of a missing measurement
    # rather than a low one.
    state, _ = run([context(dense_score=None)], policy=g.POLICY_THRESHOLD, threshold=0.75)
    assert state["decision"] == "answer"
    assert state["reason"] == "no dense score available"


# ------------------------------------------------------------------ guards

def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown policy"):
        run([context()], policy="vibes")


def test_top_dense_score_ignores_missing_values():
    assert g.top_dense_score([]) is None
    assert g.top_dense_score([context(dense_score=None)]) is None
    assert g.top_dense_score(
        [context("a", dense_score=None), context("b", dense_score=0.4)]
    ) == 0.4


def test_the_api_does_not_import_the_graph():
    """langgraph is optional, so the service must not depend on it.

    Checked as a source property rather than by import, because importing
    src.api in an environment that happens to have langgraph installed would
    pass either way.
    """
    from pathlib import Path

    api = Path(__file__).resolve().parents[1] / "src" / "api.py"
    source = api.read_text(encoding="utf-8")

    assert "src.graph" not in source
    assert "langgraph" not in source
