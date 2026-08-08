"""
Phase 3 — Evaluate the RAG pipeline with RAGAS.

RAGAS scores a RAG system on metrics that don't need a human:
  - faithfulness:        is the answer supported by the retrieved context? (no hallucination)
  - answer_relevancy:    does the answer actually address the question?
  - context_precision:   were the retrieved chunks relevant?
  - context_recall:      did we retrieve everything needed (vs the ground truth)?

RAGAS uses an LLM as a judge. Set an OpenAI key, OR configure a local judge (see docs).

Run:
    pip install -r eval/requirements.txt
    python eval/ragas_eval.py
"""
import json

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from src.rag import retrieve, build_prompt, generate


def load_testset(path: str = "data/qa_testset.jsonl") -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_eval_dataset() -> Dataset:
    """Run our RAG on each test question and collect what RAGAS needs."""
    questions, answers, contexts, ground_truths = [], [], [], []
    for row in load_testset():
        q = row["question"]
        ctx = retrieve(q)
        ctx_texts = [c["text"] for c in ctx]
        ans = generate(build_prompt(q, ctx))

        questions.append(q)
        answers.append(ans)
        contexts.append(ctx_texts)
        ground_truths.append(row["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def main():
    print("Running RAG over the test set (this calls your LLM for each question)...")
    ds = build_eval_dataset()
    print("Scoring with RAGAS...")
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    print("\n=== RAGAS results ===")
    print(result)
    # Save a report you can screenshot for the README / CV evidence
    df = result.to_pandas()
    df.to_csv("eval/ragas_report.csv", index=False)
    print("\nSaved detailed report to eval/ragas_report.csv")


if __name__ == "__main__":
    main()
