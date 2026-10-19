"""Measure both abstention policies by running them through the LangGraph flow.

This is the check that the graph in `src/graph.py` decides what the straight-line
code decides. `eval/abstention_threshold.json` already measured the threshold
rule directly; if routing it through four nodes and a conditional edge changes
the numbers, the graph is not a faithful restatement and the difference is a
defect rather than a finding.

    python -m eval.graph_abstention                    # prints both policies
    python -m eval.graph_abstention --out eval/graph_abstention.json

No LLM is involved. The decision is a pure function of retrieval, so the
generator passed into the graph raises if it is ever called - an abstention that
paid for a generation first would be a bug, and this is the cheapest way to
assert it never happens across all 50 questions.

Retrieval does run for real, against whatever `QDRANT_URL` points at, because
the decision reads retrieval's output.

## What is being counted

A question is *answerable* or *unanswerable* in the ground truth. A policy either
*answers* or *abstains*. The positive class is *answerable*, which is the
convention eval/abstention_threshold.json uses:

  sensitivity  = answered  / answerable        (did not block what it should not)
  specificity  = abstained / unanswerable      (caught what it should)

Reported as balanced accuracy and Youden's J, which is what the earlier study
used, so the two are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from eval.ground_truth import load_questions, question_set_hash
from src.corpus import corpus_hash, load_corpus


def _never_generate(prompt: str) -> str:
    raise AssertionError(
        "the generator was called during an abstention evaluation; the decision "
        "nodes must not depend on generation"
    )


@dataclass
class PolicyOutcome:
    """Counts and rates for one policy on one split."""

    policy: str
    split: str
    threshold: float | None
    answerable: int = 0
    unanswerable: int = 0
    abstained_on_unanswerable: int = 0
    answered_on_answerable: int = 0
    false_abstentions: int = 0
    false_answers: int = 0
    decisions: list[dict] = field(default_factory=list)

    # The positive class is *answerable*, matching eval/abstention_threshold.json
    # so the two files can be compared field by field. Defining it the other way
    # round leaves balanced accuracy and Youden's J unchanged, because both are
    # symmetric under swapping the classes, and silently transposes sensitivity
    # against specificity - which looks like agreement on the headline numbers
    # while disagreeing on every row.
    @property
    def sensitivity(self) -> float:
        """Answerable questions that were answered."""
        return self.answered_on_answerable / self.answerable if self.answerable else 0.0

    @property
    def specificity(self) -> float:
        """Unanswerable questions that were abstained on."""
        return self.abstained_on_unanswerable / self.unanswerable if self.unanswerable else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return (self.sensitivity + self.specificity) / 2

    @property
    def youden_j(self) -> float:
        return self.sensitivity + self.specificity - 1

    def as_dict(self, include_decisions: bool = False) -> dict:
        out = {
            "policy": self.policy,
            "split": self.split,
            "threshold": self.threshold,
            "answerable": self.answerable,
            "unanswerable": self.unanswerable,
            "sensitivity": round(self.sensitivity, 4),
            "specificity": round(self.specificity, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "youden_j": round(self.youden_j, 4),
            "false_abstentions": self.false_abstentions,
            "false_answers": self.false_answers,
        }
        if include_decisions:
            out["decisions"] = self.decisions
        return out


def evaluate(policy: str, split: str, threshold: float | None, questions) -> PolicyOutcome:
    from src import graph as g

    # decide_only stops the graph after the decision, so _never_generate is a
    # real assertion rather than a hope: an 'answer' decision would otherwise
    # route into generate and call it legitimately.
    compiled = g.build_graph(generate=_never_generate, decide_only=True)
    outcome = PolicyOutcome(policy=policy, split=split, threshold=threshold)

    for question in questions:
        if question.split != split:
            continue

        state = compiled.invoke(
            {"question": question.question, "policy": policy, "threshold": threshold}
        )
        decision = state["decision"]

        if question.answerable:
            outcome.answerable += 1
            if decision == "answer":
                outcome.answered_on_answerable += 1
            else:
                outcome.false_abstentions += 1
        else:
            outcome.unanswerable += 1
            if decision == "abstain":
                outcome.abstained_on_unanswerable += 1
            else:
                outcome.false_answers += 1

        outcome.decisions.append({
            "question_id": question.question_id,
            "answerable": question.answerable,
            "decision": decision,
            "reason": state.get("reason", ""),
            "contexts": len(state.get("contexts") or []),
            "top_dense_score": g.top_dense_score(state.get("contexts") or []),
        })

    return outcome


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="override the dev-selected threshold for the threshold policy",
    )
    args = parser.parse_args(argv)

    from src import graph as g

    questions = load_questions()
    threshold = args.threshold if args.threshold is not None else g.DEV_SELECTED_THRESHOLD

    outcomes = []
    for split in ("dev", "test"):
        for policy in g.POLICIES:
            # The no-evidence policy has no threshold, so recording one against
            # it would imply it reads a value it never looks at.
            used = threshold if policy == g.POLICY_THRESHOLD else None
            outcomes.append(evaluate(policy, split, used, questions))

    for outcome in outcomes:
        print(
            f"{outcome.split:>5} {outcome.policy:>12}  "
            f"sens {outcome.sensitivity:.4f}  spec {outcome.specificity:.4f}  "
            f"bal_acc {outcome.balanced_accuracy:.4f}  J {outcome.youden_j:+.4f}  "
            f"(false abstentions {outcome.false_abstentions}, "
            f"false answers {outcome.false_answers})"
        )

    if args.out:
        documents = load_corpus()
        record = {
            "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "environment": {
                "python": ".".join(str(n) for n in sys.version_info[:3]),
                "langgraph": _langgraph_version(),
            },
            "note": (
                "Both policies run through src/graph.py compiled with decide_only=True, "
                "so the graph ends at the decision node. The generator passed in raises "
                "if called, and was not called."
            ),
            "corpus_hash": corpus_hash(documents),
            "question_set_hash": question_set_hash(questions),
            "threshold": threshold,
            "outcomes": [o.as_dict(include_decisions=True) for o in outcomes],
        }
        args.out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")

    return 0


def _langgraph_version() -> str:
    from importlib.metadata import version

    return version("langgraph")


if __name__ == "__main__":
    raise SystemExit(main())
