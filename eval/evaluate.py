"""
Evaluate the RAG pipeline with a local LLM-as-judge (no API key needed).

These are **custom judge metrics**, not RAGAS metrics. Each is a single holistic 0-1 score
produced by one prompt to a local model. RAGAS decomposes answers into claims and weights
context precision by rank; this does neither. The names are prefixed `judge_` so they
cannot be mistaken for library implementations. See eval/README.md.

  - judge_faithfulness       : is the answer supported by the retrieved context?
  - judge_answer_relevancy   : does the answer address the question?
  - judge_context_precision  : are the retrieved chunks relevant? (see caveat below)
  - judge_answer_correctness : does the answer match the reference answer?

`judge_context_precision` is retained per-question but is NOT a retrieval metric: the judge
receives every chunk concatenated, so ranking is invisible to it. It has scored a flat 0.80
on every question recorded so far. Real retrieval metrics need relevance labels, which this
test set does not have.

Run:
    python -m eval.evaluate
"""
import csv
import json
import math
import re
import statistics

import requests

from src.config import cfg
from src.rag import build_prompt, generate, retrieve

JUDGE_MODEL = cfg.OLLAMA_MODEL

METRICS = [
    "judge_faithfulness",
    "judge_answer_relevancy",
    "judge_context_precision",
    "judge_answer_correctness",
]

_SCORE_PATTERN = re.compile(r"[01](?:\.\d+)?")


def parse_judge_score(text: str) -> float:
    """Pull a 0..1 score out of the judge's reply, or NaN if there isn't one.

    Kept separate from the HTTP call so the parsing rules can be tested without a model:
    small judges pad their answers, refuse, or return prose, and each of those has to
    degrade to NaN rather than to a wrong number.
    """
    match = _SCORE_PATTERN.search(text or "")
    if not match:
        return float("nan")
    return max(0.0, min(1.0, float(match.group(0))))


def average_scores(rows: list[dict], metrics: list[str]) -> dict[str, dict]:
    """Average each metric, reporting how many questions actually contributed.

    A judge that fails to answer yields NaN. Those rows are excluded from the mean, so the
    denominator can silently shrink - an average over four questions once looked identical
    to an average over ten. `used` and `dropped` are returned alongside the mean so a
    partially-failed run is visible instead of merely optimistic.
    """
    summary = {}
    for metric in metrics:
        values = [r[metric] for r in rows if not math.isnan(r[metric])]
        summary[metric] = {
            "mean": statistics.mean(values) if values else float("nan"),
            "used": len(values),
            "dropped": len(rows) - len(values),
        }
    return summary


def _judge(prompt: str) -> float:
    """Ask the judge LLM for a single 0..1 score."""
    full = (
        prompt
        + "\n\nGive ONLY a single decimal number from 0.0 to 1.0 (nothing else).\nScore:"
    )
    resp = requests.post(
        f"{cfg.OLLAMA_URL}/api/generate",
        json={
            "model": JUDGE_MODEL,
            "prompt": full,
            "stream": False,
            # num_predict keeps the answer to a few tokens => just the number
            "options": {"temperature": 0, "num_predict": 6},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return parse_judge_score(resp.json()["response"].strip())


def judge_faithfulness(answer: str, contexts: list[str]) -> float:
    ctx = "\n---\n".join(contexts)
    return _judge(
        f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}\n\n"
        "Score how well the ANSWER is supported by the CONTEXT "
        "(1 = every claim is supported, 0 = not supported / invented)."
    )


def judge_answer_relevancy(question: str, answer: str) -> float:
    return _judge(
        f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\n"
        "Score how directly the ANSWER addresses the QUESTION "
        "(1 = fully on-point, 0 = irrelevant)."
    )


def judge_context_precision(question: str, contexts: list[str]) -> float:
    ctx = "\n---\n".join(contexts)
    return _judge(
        f"QUESTION:\n{question}\n\nRETRIEVED CONTEXT:\n{ctx}\n\n"
        "Score how relevant the RETRIEVED CONTEXT is for answering the QUESTION "
        "(1 = highly relevant, 0 = irrelevant)."
    )


def judge_answer_correctness(answer: str, ground_truth: str) -> float:
    return _judge(
        f"REFERENCE ANSWER:\n{ground_truth}\n\nMODEL ANSWER:\n{answer}\n\n"
        "Score how factually consistent the MODEL ANSWER is with the REFERENCE ANSWER "
        "(1 = same meaning, 0 = contradicts / wrong)."
    )


def load_testset(path: str = "data/qa_testset.jsonl") -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main(limit: int | None = None):
    rows = load_testset()
    if limit:
        rows = rows[:limit]

    per_row = []
    print(f"Evaluating {len(rows)} questions (judge = {JUDGE_MODEL}, also the generator)...\n")
    for i, row in enumerate(rows, 1):
        q, gt = row["question"], row["ground_truth"]
        ctx = retrieve(q)
        ctx_texts = [c["text"] for c in ctx]
        ans = generate(build_prompt(q, ctx))

        scores = {
            "judge_faithfulness": judge_faithfulness(ans, ctx_texts),
            "judge_answer_relevancy": judge_answer_relevancy(q, ans),
            "judge_context_precision": judge_context_precision(q, ctx_texts),
            "judge_answer_correctness": judge_answer_correctness(ans, gt),
        }
        per_row.append({"question": q, "answer": ans, **scores})
        print(f"[{i}/{len(rows)}] {q[:55]:<55}  "
              + "  ".join(f"{k.replace('judge_', '')[:4]}={v:.2f}" for k, v in scores.items()))

    summary = average_scores(per_row, METRICS)
    print("\n=== Average scores ===")
    for metric, stats in summary.items():
        note = f"  ({stats['dropped']} unscored)" if stats["dropped"] else ""
        print(f"  {metric:<26} {stats['mean']:.3f}  [n={stats['used']}]{note}")
    print("\nNote: these describe answer quality only. Retrieval quality is not measured -")
    print("the test set has no relevance labels. See eval/README.md.")

    with open("eval/eval_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question", "answer", *METRICS])
        w.writeheader()
        w.writerows(per_row)
    print("\nSaved per-question scores to eval/eval_report.csv")
    return summary


if __name__ == "__main__":
    main()
