"""
Phase 3 - Evaluate the RAG pipeline with an LLM-as-judge (local, free, no API key).

These are the standard RAG-evaluation metrics (the same ones the RAGAS library
implements). We score them with a local LLM via Ollama, so it runs fully offline:

  - faithfulness       : is the answer supported by the retrieved context? (no hallucination)
  - answer_relevancy   : does the answer actually address the question?
  - context_precision  : are the retrieved chunks relevant to the question?
  - answer_correctness : does the answer match the reference (ground-truth) answer?

Each metric is scored 0.0 - 1.0 by the judge LLM; we report the average over the test set.

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


def _judge(prompt: str) -> float:
    """Ask the judge LLM for a single 0..1 score and parse it robustly."""
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
    text = resp.json()["response"].strip()
    m = re.search(r"[01](?:\.\d+)?", text)
    if not m:
        return float("nan")
    return max(0.0, min(1.0, float(m.group(0))))


def faithfulness(answer: str, contexts: list[str]) -> float:
    ctx = "\n---\n".join(contexts)
    return _judge(
        f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}\n\n"
        "Score how well the ANSWER is supported by the CONTEXT "
        "(1 = every claim is supported, 0 = not supported / invented)."
    )


def answer_relevancy(question: str, answer: str) -> float:
    return _judge(
        f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\n"
        "Score how directly the ANSWER addresses the QUESTION "
        "(1 = fully on-point, 0 = irrelevant)."
    )


def context_precision(question: str, contexts: list[str]) -> float:
    ctx = "\n---\n".join(contexts)
    return _judge(
        f"QUESTION:\n{question}\n\nRETRIEVED CONTEXT:\n{ctx}\n\n"
        "Score how relevant the RETRIEVED CONTEXT is for answering the QUESTION "
        "(1 = highly relevant, 0 = irrelevant)."
    )


def answer_correctness(answer: str, ground_truth: str) -> float:
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
    print(f"Evaluating {len(rows)} questions (judge = {JUDGE_MODEL})...\n")
    for i, row in enumerate(rows, 1):
        q, gt = row["question"], row["ground_truth"]
        ctx = retrieve(q)
        ctx_texts = [c["text"] for c in ctx]
        ans = generate(build_prompt(q, ctx))

        scores = {
            "faithfulness": faithfulness(ans, ctx_texts),
            "answer_relevancy": answer_relevancy(q, ans),
            "context_precision": context_precision(q, ctx_texts),
            "answer_correctness": answer_correctness(ans, gt),
        }
        per_row.append({"question": q, "answer": ans, **scores})
        print(f"[{i}/{len(rows)}] {q[:55]:<55}  "
              + "  ".join(f"{k[:4]}={v:.2f}" for k, v in scores.items()))

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "answer_correctness"]
    print("\n=== Average scores ===")
    averages = {}
    for m in metrics:
        vals = [r[m] for r in per_row if not math.isnan(r[m])]  # drop NaN
        avg = statistics.mean(vals) if vals else float("nan")
        averages[m] = avg
        print(f"  {m:<20} {avg:.3f}")

    with open("eval/eval_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question", "answer", *metrics])
        w.writeheader()
        w.writerows(per_row)
    print("\nSaved per-question scores to eval/eval_report.csv")
    return averages


if __name__ == "__main__":
    main()
