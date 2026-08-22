"""
Classify retrieval failures from a reference run.

    python -m eval.failure_analysis --run eval/reference_run_nfip.json

Reads a run produced by `eval.reference_run` and buckets every miss. Reads only the JSON -
it does not re-query the retriever, so the analysis is reproducible from the committed
artefact.

The point is diagnosis, not a score. "Hit rate is 0.61" says something is wrong; "the
retriever returned the right provision from the wrong form in most misses" says what.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WRONG_FORM = "wrong near-duplicate form"
RIGHT_DOC_WRONG_CHUNK = "correct document, wrong chunk"
MULTI_CHUNK_PARTIAL = "multi-chunk partial miss"
IRRELEVANT = "irrelevant retrieval"
UNANSWERABLE_HIGH_SIM = "unanswerable but high similarity"
OTHER = "other"

#: Questions whose evidence turns on a figure or a period; a miss here is worth separating
#: because the retriever plausibly matched the topic and missed the number.
NUMERIC_CATEGORIES = {"numeric_limit", "time_period"}
EXCLUSION_CATEGORIES = {"exclusion"}


def classify(record: dict) -> str:
    """Bucket one question's retrieval outcome."""
    relevant = set(record["relevant_chunk_ids"])
    retrieved = record["retrieved_chunk_ids"]

    if not record["answerable"]:
        return UNANSWERABLE_HIGH_SIM

    if any(chunk_id in relevant for chunk_id in retrieved):
        # Something relevant was found. Only a partial multi-chunk result is a failure.
        if len(relevant) > 1 and not relevant.issubset(set(retrieved)):
            return MULTI_CHUNK_PARTIAL
        return ""  # not a failure

    relevant_documents = set(record["relevant_document_ids"])
    retrieved_documents = record["retrieved_document_ids"]

    if not retrieved_documents:
        return IRRELEVANT if not retrieved else OTHER

    # Judge on the TOP-1 document, not on the set. With three near-duplicate forms the
    # top-5 almost always contains a chunk from every form, so an intersection test would
    # report "right document" for nearly every miss and hide the actual failure.
    if retrieved_documents[0] not in relevant_documents:
        return WRONG_FORM

    if record["category"] in NUMERIC_CATEGORIES:
        return f"{RIGHT_DOC_WRONG_CHUNK} (numeric/period)"
    if record["category"] in EXCLUSION_CATEGORIES:
        return f"{RIGHT_DOC_WRONG_CHUNK} (exclusion)"
    return RIGHT_DOC_WRONG_CHUNK


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="eval/reference_run_nfip.json")
    args = parser.parse_args()

    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    records = run["per_question"]

    failures = []
    for record in records:
        bucket = classify(record)
        if bucket and bucket != UNANSWERABLE_HIGH_SIM:
            failures.append((bucket, record))

    counts: dict[str, int] = {}
    for bucket, _ in failures:
        counts[bucket] = counts.get(bucket, 0) + 1

    answerable = [r for r in records if r["answerable"]]
    print(f"Run: {args.run}  split={run['run']['split']}")
    print(f"Answerable questions: {len(answerable)}   failures: {len(failures)}\n")

    print("Failure classes")
    print("-" * 62)
    for bucket, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        share = count / len(answerable)
        print(f"  {bucket:<44} {count:>2}  ({share:.0%})")

    print("\nFailed questions")
    print("-" * 100)
    print(f"  {'id':<10} {'category':<16} {'asked about':<28} {'top hit came from':<28}")
    print("-" * 100)
    for bucket, record in sorted(failures, key=lambda f: f[1]["question_id"]):
        asked = ", ".join(d.replace("nfip-sfip-", "") for d in record["relevant_document_ids"])
        got = ", ".join(
            d.replace("nfip-sfip-", "") for d in record["retrieved_document_ids"][:2]
        )
        print(f"  {record['question_id']:<10} {record['category']:<16} {asked:<28} {got:<28}")

    unanswerable = [r for r in records if not r["answerable"]]
    if unanswerable:
        print("\nUnanswerable questions (no relevant chunk exists; similarity is descriptive)")
        print("-" * 100)
        for record in sorted(unanswerable, key=lambda r: -(r["top_score"] or 0)):
            top = record["retrieved_document_ids"][0].replace("nfip-sfip-", "") if \
                record["retrieved_document_ids"] else "-"
            print(f"  {record['question_id']:<10} top_score={record['top_score']:.4f}  "
                  f"nearest={top:<22} {record['question'][:52]}")
        scores = [r["top_score"] for r in unanswerable if r["top_score"]]
        answerable_scores = [r["top_score"] for r in answerable if r["top_score"]]
        print(f"\n  mean top-1 similarity: unanswerable {sum(scores)/len(scores):.4f} "
              f"vs answerable {sum(answerable_scores)/len(answerable_scores):.4f}")
        print("  A separable gap would justify an abstention threshold. Deciding that needs "
              "a\n  threshold study on the dev split; none has been run, so none is proposed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
