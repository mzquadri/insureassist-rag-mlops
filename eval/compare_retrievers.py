"""
Compare retrieval architectures on the DEV split.

    python -m eval.compare_retrievers --out eval/dev_comparison.json

DEV only, by design. Every architecture and configuration decision is made here; the TEST
split is not touched until one configuration has been frozen. Running this against TEST
would be tuning on the held-out set.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval.ground_truth import load_questions
from eval.metrics import (
    hit_rate_at_k,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.config import cfg
from src.corpus import chunk_corpus, load_corpus
from src.providers import get_embedder, get_vector_store
from src.retrieval import BM25Index, Hit, reciprocal_rank_fusion

K_VALUES = (1, 3, 5)


def dense_search(question: str, limit: int, collection: str) -> list[Hit]:
    embedder = get_embedder()
    store = get_vector_store()
    vector = embedder.encode(
        cfg.BGE_QUERY_PREFIX + question, normalize_embeddings=True
    ).tolist()
    result = store.query_points(
        collection_name=collection, query=vector, limit=limit, with_payload=True
    )
    return [Hit(p.payload["chunk_id"], p.score) for p in result.points]


def score_questions(questions, retrieve, top_k: int) -> dict:
    """Run a retriever over the answerable questions and aggregate."""
    rows = []
    latencies = []
    for question in questions:
        if not question.answerable:
            continue
        started = time.perf_counter()
        hits = retrieve(question.question, top_k)
        latencies.append((time.perf_counter() - started) * 1000)

        ids = [h.chunk_id for h in hits]
        relevant = set(question.relevant_chunk_ids)
        documents = []
        for chunk_id in ids:
            document_id = chunk_id.split("#")[0]
            if document_id not in documents:
                documents.append(document_id)

        rows.append({
            "question_id": question.question_id,
            "category": question.category,
            "relevant_documents": question.relevant_document_ids,
            "top_document": documents[0] if documents else None,
            "top_document_correct": bool(
                documents and documents[0] in question.relevant_document_ids
            ),
            **{f"hit_rate@{k}": hit_rate_at_k(ids, relevant, k) for k in K_VALUES},
            **{f"recall@{k}": recall_at_k(ids, relevant, k) for k in K_VALUES},
            **{f"precision@{k}": precision_at_k(ids, relevant, k) for k in K_VALUES},
            "reciprocal_rank": reciprocal_rank(ids, relevant),
        })

    metrics = {}
    for k in K_VALUES:
        for name in ("hit_rate", "recall", "precision"):
            metrics[f"{name}@{k}"] = round(mean(r[f"{name}@{k}"] for r in rows), 4)
    metrics["mrr"] = round(mean(r["reciprocal_rank"] for r in rows), 4)
    metrics["top_document_accuracy"] = round(
        mean(float(r["top_document_correct"]) for r in rows), 4
    )

    by_category: dict[str, dict] = {}
    for row in rows:
        bucket = by_category.setdefault(row["category"], {"n": 0, "hit": [], "mrr": []})
        bucket["n"] += 1
        bucket["hit"].append(row["hit_rate@5"])
        bucket["mrr"].append(row["reciprocal_rank"])
    by_category = {
        name: {
            "n": b["n"],
            "hit_rate@5": round(mean(b["hit"]), 4),
            "mrr": round(mean(b["mrr"]), 4),
        }
        for name, b in sorted(by_category.items())
    }

    return {
        "metrics": metrics,
        "by_category": by_category,
        "latency_ms_mean": round(mean(latencies), 2),
        "per_question": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="eval/dev_comparison.json")
    parser.add_argument("--collection", default=cfg.QDRANT_COLLECTION)
    args = parser.parse_args()

    documents = load_corpus()
    dev = [q for q in load_questions() if q.split == "dev"]
    answerable = [q for q in dev if q.answerable]
    print(f"DEV: {len(dev)} questions, {len(answerable)} answerable\n")

    results: dict[str, dict] = {}

    # --- Architectures at the current chunking -----------------------------------------
    chunks = chunk_corpus(documents, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
    bm25 = BM25Index(chunks)

    def bm25_search(question, limit):
        return bm25.search(question, limit)

    def dense(question, limit):
        return dense_search(question, limit, args.collection)

    def hybrid(question, limit):
        return reciprocal_rank_fusion(
            [dense(question, limit * 2), bm25_search(question, limit * 2)], limit=limit
        )

    for name, fn in (("dense", dense), ("bm25", bm25_search), ("hybrid_rrf", hybrid)):
        results[name] = score_questions(dev, fn, 5)
        m = results[name]["metrics"]
        print(f"  {name:12} hit@1 {m['hit_rate@1']:.3f}  hit@5 {m['hit_rate@5']:.3f}  "
              f"mrr {m['mrr']:.3f}  topdoc {m['top_document_accuracy']:.3f}")

    # --- BM25 parameter check ----------------------------------------------------------
    # A small, recorded search space. Defaults are kept unless something clearly beats them.
    print("\n  BM25 parameter sweep (dev):")
    sweep = {}
    for k1 in (0.9, 1.2, 1.5, 2.0):
        for b in (0.5, 0.75, 1.0):
            index = BM25Index(chunks, k1=k1, b=b)
            scored = score_questions(dev, lambda q, n, i=index: i.search(q, n), 5)
            sweep[f"k1={k1},b={b}"] = scored["metrics"]
            print(f"    k1={k1:<4} b={b:<5} hit@5 {scored['metrics']['hit_rate@5']:.3f}  "
                  f"mrr {scored['metrics']['mrr']:.3f}")
    results["bm25_parameter_sweep"] = sweep

    # --- top_k -------------------------------------------------------------------------
    print("\n  top_k (dev, best architecture so far):")
    topk = {}
    for k in (3, 5, 8, 10):
        scored = score_questions(dev, hybrid, k)
        topk[str(k)] = scored["metrics"]
        print(f"    top_k={k:<3} hit@5 {scored['metrics']['hit_rate@5']:.3f}  "
              f"mrr {scored['metrics']['mrr']:.3f}")
    results["hybrid_top_k"] = topk

    Path(args.out).write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
