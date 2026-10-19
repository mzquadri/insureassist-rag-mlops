"""The ask flow as an explicit LangGraph state machine: retrieve, decide, then answer or abstain.

`rag.answer()` already does this. It is a straight-line function with the
abstention folded into an `if not contexts` early return, and for a three-step
flow that is the right shape - this module does not replace it and the API does
not call it.

What it adds is one thing: the decision becomes a node with a name, so the
*policy* can be swapped and measured without touching retrieval or generation.
That is useful here specifically because which abstention policy to ship was a
real question in this repository, it was answered with evidence, and the answer
was counter-intuitive.

## The decision this encodes

`eval/abstention_threshold.json` tested whether a similarity threshold is a
defensible abstention rule. Choosing the threshold on dev by Youden's J and
applying it unchanged to test:

| | dev | test |
|---|---|---|
| balanced accuracy | 0.8016 | 0.5556 |
| Youden's J | 0.6032 | 0.1111 |

Always answering scores 0.5 balanced accuracy by construction. So on the held-out
split the threshold buys 0.0556 over doing nothing, having looked worth 0.30 on
dev. The reason is in the same file: **17 of 18 unanswerable questions score above
the weakest answerable one**, so the signal barely separates the two classes.

`POLICY_NO_EVIDENCE` is therefore what ships, and `POLICY_THRESHOLD` exists to
keep the rejected alternative runnable rather than only described.
`eval/graph_abstention.py` runs both through this graph and reproduces the table
above, which is also what checks the graph against the function it mirrors.

## Why generation is injected

`generate` is a constructor argument. The decision nodes are a pure function of
retrieval, so the evaluation runs the graph with a generator that is never
called, and measures abstain-or-answer over all 50 questions without an LLM
anywhere. Reaching into `rag.generate` directly would have made the eval need
Ollama to measure a decision that does not involve it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, NotRequired, TypedDict

from src import rag

#: Abstain only when retrieval returned nothing at all. What the service ships.
POLICY_NO_EVIDENCE = "no_evidence"
#: Abstain when the top-1 dense score is below a threshold. Measured and rejected.
POLICY_THRESHOLD = "threshold"

POLICIES = (POLICY_NO_EVIDENCE, POLICY_THRESHOLD)

#: The value eval/abstention_threshold.json selected on dev by Youden's J.
#: Carried here so the rejected policy is reproducible, not because it is good.
DEV_SELECTED_THRESHOLD = 0.739515735


class AskState(TypedDict):
    """What flows between nodes.

    `question` and `policy` are required: they are the input, and a run without
    them is a programming error rather than a state to handle. Everything else
    is NotRequired because the nodes fill it in progressively - `contexts` by
    retrieval, `decision` by the decision node, and the answer fields by
    whichever terminal node ran.
    """

    question: str
    policy: str
    threshold: NotRequired[float | None]
    contexts: NotRequired[list[dict]]
    decision: NotRequired[Literal["answer", "abstain"]]
    reason: NotRequired[str]
    status: NotRequired[str]
    answer: NotRequired[str]
    citations: NotRequired[list[dict]]


def top_dense_score(contexts: list[dict]) -> float | None:
    """The signal the threshold policy reads, or None when it is unavailable.

    `dense_score` is absent when a chunk reached the fused list from BM25 only,
    so `max` is taken over the scores that exist rather than defaulting a
    missing one to 0.0 - which would read as "definitely irrelevant" for a
    chunk that simply was not scored by the dense retriever.
    """
    scores = [c["dense_score"] for c in contexts if c.get("dense_score") is not None]
    return max(scores) if scores else None


def build_graph(
    generate: Callable[[str], str] | None = None,
    retrieve: Callable[[str], list[dict]] | None = None,
    decide_only: bool = False,
):
    """Compile the flow. Both dependencies are injectable so the eval needs no LLM.

    `decide_only` stops the graph after the decision node instead of routing on
    it, so `retrieve -> decide -> END`. That is what `eval/graph_abstention.py`
    measures: it scores the decision over 50 questions and has no use for
    generated text. Without it, an "answer" decision correctly routes into
    `generate`, and an evaluation of the *decision* would be paying an LLM to
    produce output it discards. The conditional routing itself is covered by
    tests/test_graph.py rather than by the evaluation.

    Raises ImportError with an actionable message when langgraph is absent,
    rather than at import time: the service does not use this module, and a
    missing optional dependency should not stop `src.api` from starting.
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:  # pragma: no cover - exercised by the import guard test
        raise ImportError(
            "langgraph is not installed. It is optional and only this module needs it: "
            "pip install -r requirements-graph.txt"
        ) from exc

    _retrieve = retrieve or rag.retrieve
    _generate = generate or rag.generate

    def retrieve_node(state: AskState) -> dict[str, Any]:
        return {"contexts": _retrieve(state["question"])}

    def decide_node(state: AskState) -> dict[str, Any]:
        """Set the decision and the reason for it. Does not act on either."""
        contexts = state.get("contexts") or []
        policy = state.get("policy", POLICY_NO_EVIDENCE)

        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; expected one of {list(POLICIES)}")

        if not contexts:
            # Common to both policies, and the only case the dev evidence
            # actually supports: there is no evidence, so there is nothing to
            # answer from.
            return {"decision": "abstain", "reason": "no contexts retrieved"}

        if policy == POLICY_NO_EVIDENCE:
            return {"decision": "answer", "reason": "contexts retrieved"}

        threshold = state.get("threshold")
        if threshold is None:
            threshold = DEV_SELECTED_THRESHOLD
        top = top_dense_score(contexts)

        if top is None:
            # No dense score to judge by. Answering is the conservative choice
            # here: the alternative is abstaining because of a missing
            # measurement rather than because of a low one.
            return {"decision": "answer", "reason": "no dense score available"}

        if top < threshold:
            return {
                "decision": "abstain",
                "reason": f"top dense score {top:.4f} below threshold {threshold:.4f}",
            }
        return {
            "decision": "answer",
            "reason": f"top dense score {top:.4f} at or above threshold {threshold:.4f}",
        }

    def generate_node(state: AskState) -> dict[str, Any]:
        contexts = state["contexts"]
        text = _generate(rag.build_prompt(state["question"], contexts))
        return {
            "status": rag.STATUS_ANSWERED,
            "answer": text,
            "citations": [rag.citation(c) for c in contexts],
        }

    def abstain_node(_state: AskState) -> dict[str, Any]:
        return {
            "status": rag.STATUS_INSUFFICIENT_EVIDENCE,
            "answer": (
                "No policy text was retrieved for this question, so it cannot be answered "
                "from the indexed documents."
            ),
            "citations": [],
        }

    graph = StateGraph(AskState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("decide", decide_node)
    graph.add_node("generate", generate_node)
    graph.add_node("abstain", abstain_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "decide")

    if decide_only:
        graph.add_edge("decide", END)
    else:
        # The conditional edge is the whole reason this is a graph. `decide`
        # records a decision; this edge is what acts on it, and the two are
        # deliberately separate so the decision is inspectable without having
        # run the branch. `decide_only` above is that separation used.
        graph.add_conditional_edges(
            "decide",
            lambda state: state["decision"],
            {"answer": "generate", "abstain": "abstain"},
        )
        graph.add_edge("generate", END)
        graph.add_edge("abstain", END)

    return graph.compile()


def ask(
    question: str,
    policy: str = POLICY_NO_EVIDENCE,
    threshold: float | None = None,
    **dependencies: Any,
) -> AskState:
    """Run the flow once. Returns the full final state, not just the answer.

    The state is returned whole because `decision` and `reason` are the point:
    a caller that only wanted the text would use `rag.answer()`.
    """
    compiled = build_graph(**dependencies)
    return compiled.invoke(
        {
            "question": question,
            "policy": policy,
            "threshold": threshold,
        }
    )
