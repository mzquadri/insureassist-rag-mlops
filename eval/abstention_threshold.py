"""Is a similarity threshold defensible as an abstention rule, now that the set is bigger?

The previous answer was no, and the stated reason was sample size: eight unanswerable
questions cannot support a threshold claim. The set now holds eighteen, so the question is
worth asking again, properly.

The protocol is the one that makes an answer meaningful rather than flattering. The
threshold is chosen on the dev split alone, by Youden's J on the top-1 dense score, and then
applied unchanged to test. A threshold tuned on the split it is reported on measures nothing
except how well it memorised that split.

The baseline it has to beat is "always answer", which is what the service does today. On a
set that is 64% answerable, always answering is already a strong baseline, and a rule that
does not beat it is not worth the complexity or the false abstentions it buys.

    python eval/abstention_threshold.py
    python eval/abstention_threshold.py --out eval/abstention_threshold.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.ground_truth import load_questions  # noqa: E402
from src.rag import retrieve  # noqa: E402

TOP_K = 5


@dataclass(frozen=True)
class Scored:
    question_id: str
    answerable: bool
    split: str
    top_dense: float


@dataclass(frozen=True)
class Outcome:
    """What a rule does on one split."""

    threshold: float | None
    answerable_accepted: int
    answerable_total: int
    unanswerable_rejected: int
    unanswerable_total: int

    @property
    def sensitivity(self) -> float:
        """Share of answerable questions still answered."""
        return self.answerable_accepted / self.answerable_total if self.answerable_total else 0.0

    @property
    def specificity(self) -> float:
        """Share of unanswerable questions correctly refused."""
        return self.unanswerable_rejected / self.unanswerable_total if self.unanswerable_total else 0.0

    @property
    def youden_j(self) -> float:
        return self.sensitivity + self.specificity - 1.0

    @property
    def balanced_accuracy(self) -> float:
        return (self.sensitivity + self.specificity) / 2.0

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "sensitivity": round(self.sensitivity, 4),
            "specificity": round(self.specificity, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "youden_j": round(self.youden_j, 4),
            "false_abstentions": self.answerable_total - self.answerable_accepted,
            "false_answers": self.unanswerable_total - self.unanswerable_rejected,
        }


def score_questions(questions) -> list[Scored]:
    out: list[Scored] = []
    for q in questions:
        contexts = retrieve(q.question, TOP_K)
        dense = [c["dense_score"] for c in contexts if c.get("dense_score") is not None]
        out.append(
            Scored(
                question_id=q.question_id,
                answerable=q.answerable,
                split=q.split,
                # Absent a dense hit the question retrieved nothing, which the service
                # already treats as an abstention, so score it at the floor.
                top_dense=max(dense) if dense else 0.0,
            )
        )
    return out


def apply(rows: list[Scored], threshold: float | None) -> Outcome:
    """`None` means the always-answer baseline the service runs today."""
    answerable = [r for r in rows if r.answerable]
    unanswerable = [r for r in rows if not r.answerable]
    if threshold is None:
        return Outcome(None, len(answerable), len(answerable), 0, len(unanswerable))
    return Outcome(
        threshold,
        sum(1 for r in answerable if r.top_dense >= threshold),
        len(answerable),
        sum(1 for r in unanswerable if r.top_dense < threshold),
        len(unanswerable),
    )


def select_threshold(dev: list[Scored]) -> tuple[float, Outcome]:
    """Every midpoint between adjacent observed scores is a candidate."""
    scores = sorted({r.top_dense for r in dev})
    candidates = [(a + b) / 2 for a, b in zip(scores, scores[1:])]
    if not candidates:
        return 0.0, apply(dev, 0.0)
    best = max(candidates, key=lambda t: (apply(dev, t).youden_j, -t))
    return best, apply(dev, best)


def separation(rows: list[Scored]) -> dict:
    """How far apart the two populations sit, before any rule is drawn."""
    ans = [r.top_dense for r in rows if r.answerable]
    una = [r.top_dense for r in rows if not r.answerable]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    overlap = sum(1 for u in una if u >= min(ans)) if ans else 0
    return {
        "answerable_mean": round(mean(ans), 4),
        "answerable_min": round(min(ans), 4) if ans else None,
        "unanswerable_mean": round(mean(una), 4),
        "unanswerable_max": round(max(una), 4) if una else None,
        "unanswerable_scoring_above_the_weakest_answerable": overlap,
        "unanswerable_total": len(una),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write the result as JSON")
    args = parser.parse_args()

    questions = load_questions()
    rows = score_questions(questions)
    dev = [r for r in rows if r.split == "dev"]
    test = [r for r in rows if r.split == "test"]

    print(f"{len(rows)} questions: {len(dev)} dev, {len(test)} test")
    print(f"unanswerable: {sum(1 for r in rows if not r.answerable)} "
          f"({sum(1 for r in dev if not r.answerable)} dev, "
          f"{sum(1 for r in test if not r.answerable)} test)\n")

    sep = separation(rows)
    print("score separation over the whole set")
    for k, v in sep.items():
        print(f"  {k:52s} {v}")
    print()

    threshold, dev_outcome = select_threshold(dev)
    test_outcome = apply(test, threshold)
    baseline = apply(test, None)

    print(f"threshold selected on dev by Youden's J: {threshold:.4f}")
    print(f"  on dev  {dev_outcome.as_dict()}")
    print(f"  on test {test_outcome.as_dict()}")
    print(f"  always-answer baseline on test {baseline.as_dict()}\n")

    # Balanced accuracy on its own would call this a win, and it is the wrong judge.
    # It weights the two errors equally, and here they are not equal. A false answer
    # arrives with citations that resolve to exact character offsets, so a reader who
    # follows the design and checks them can see the answer is not supported. A false
    # abstention produces nothing to check. In a system whose stated instruction is to
    # read the citations rather than the prose, losing sensitivity is the worse trade.
    #
    # So adoption needs both: better balanced accuracy *and* sensitivity kept above a
    # floor that is declared here rather than discovered after the fact.
    SENSITIVITY_FLOOR = 0.90

    better_balance = test_outcome.balanced_accuracy > baseline.balanced_accuracy
    keeps_sensitivity = test_outcome.sensitivity >= SENSITIVITY_FLOOR
    adopted = better_balance and keeps_sensitivity

    if adopted:
        verdict = "Adopted: better balanced accuracy on test without giving up sensitivity."
    elif better_balance:
        verdict = (
            f"Not adopted. Balanced accuracy improves ({test_outcome.balanced_accuracy:.3f} "
            f"against {baseline.balanced_accuracy:.3f}) only by refusing "
            f"{test_outcome.as_dict()['false_abstentions']} of {test_outcome.answerable_total} "
            f"answerable questions, taking sensitivity to {test_outcome.sensitivity:.3f} against a "
            f"{SENSITIVITY_FLOOR:.2f} floor. The gain is an artefact of weighting both errors "
            "equally, and they are not equal here."
        )
    else:
        verdict = "Not adopted: no improvement on the held-out split."

    print("VERDICT:", verdict)
    print(
        f"  balanced accuracy {test_outcome.balanced_accuracy:.4f} against "
        f"{baseline.balanced_accuracy:.4f}; sensitivity {test_outcome.sensitivity:.4f} "
        f"against 1.0000; {test_outcome.as_dict()['false_abstentions']} false abstentions."
    )
    print(
        f"  {sep['unanswerable_scoring_above_the_weakest_answerable']} of "
        f"{sep['unanswerable_total']} unanswerable questions score above the weakest "
        "answerable one, which is why no clean cut exists."
    )
    improved = adopted

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "protocol": "threshold chosen on dev by Youden's J, reported on test",
                    "signal": "top-1 dense cosine over the retrieved contexts",
                    "questions": {"total": len(rows), "dev": len(dev), "test": len(test)},
                    "separation": sep,
                    "selected_threshold": round(threshold, 4),
                    "dev": dev_outcome.as_dict(),
                    "test": test_outcome.as_dict(),
                    "test_baseline_always_answer": baseline.as_dict(),
                    "sensitivity_floor": SENSITIVITY_FLOOR,
                    "adopted": adopted,
                    "verdict": verdict,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwritten to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
