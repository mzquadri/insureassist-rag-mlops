"""
Dense-retrieval reference run over the NFIP benchmark.

    python -m eval.reference_run --split test --out eval/reference_run_nfip.json

Measures the existing BGE dense retriever against the labelled ground truth and writes a
machine-readable result. No BM25, no reranking, no generation - this establishes what the
current retriever actually does before anything is added to it.

Answerable and unanswerable questions are reported separately and never averaged together.
An unanswerable question has no relevant chunk, so it has no recall; folding it in as a
zero would invent a failure, and folding it in as a one would invent a success. What is
reported for those is the raw similarity of the top hit, which is descriptive only - no
abstention threshold is proposed here, because no threshold study has been run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from eval.ground_truth import load_questions, question_set_hash, summarise
from eval.metrics import (
    hit_rate_at_k,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.config import cfg
from src.corpus import chunk_corpus, config_hash, corpus_hash, load_corpus
from src.providers import get_embedder, get_vector_store

K_VALUES = (1, 3, 5)
MAX_K = max(K_VALUES)


def retrieve_chunk_ids(question: str, limit: int) -> tuple[list[str], list[float], float]:
    """Return ranked chunk IDs, their scores, and the elapsed milliseconds."""
    embedder = get_embedder()
    store = get_vector_store()

    started = time.perf_counter()
    vector = embedder.encode(
        cfg.BGE_QUERY_PREFIX + question, normalize_embeddings=True
    ).tolist()
    result = store.query_points(
        collection_name=cfg.QDRANT_COLLECTION,
        query=vector,
        limit=limit,
        with_payload=True,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    ids = [point.payload["chunk_id"] for point in result.points]
    scores = [point.score for point in result.points]
    return ids, scores, elapsed_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--out", default="eval/reference_run_nfip.json")
    parser.add_argument("--top-k", type=int, default=MAX_K)
    args = parser.parse_args()

    documents = load_corpus()
    chunks = chunk_corpus(documents, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
    all_questions = load_questions()
    questions = [
        q for q in all_questions if args.split == "all" or q.split == args.split
    ]

    answerable = [q for q in questions if q.answerable]
    unanswerable = [q for q in questions if not q.answerable]

    per_question = []
    latencies = []

    for question in questions:
        ids, scores, elapsed_ms = retrieve_chunk_ids(question.question, args.top_k)
        latencies.append(elapsed_ms)
        relevant = set(question.relevant_chunk_ids)

        retrieved_documents = []
        for chunk_id in ids:
            document_id = chunk_id.split("#")[0]
            if document_id not in retrieved_documents:
                retrieved_documents.append(document_id)

        record = {
            "question_id": question.question_id,
            "question": question.question,
            "category": question.category,
            "difficulty": question.difficulty,
            "answerable": question.answerable,
            "relevant_chunk_ids": sorted(relevant),
            "relevant_document_ids": question.relevant_document_ids,
            "retrieved_chunk_ids": ids,
            "retrieved_document_ids": retrieved_documents,
            "top_score": scores[0] if scores else None,
            "latency_ms": round(elapsed_ms, 2),
        }
        if question.answerable:
            record["metrics"] = {
                **{f"hit_rate@{k}": hit_rate_at_k(ids, relevant, k) for k in K_VALUES},
                **{f"recall@{k}": recall_at_k(ids, relevant, k) for k in K_VALUES},
                **{f"precision@{k}": precision_at_k(ids, relevant, k) for k in K_VALUES},
                "reciprocal_rank": reciprocal_rank(ids, relevant),
            }
            # Did the retriever at least reach the right form?
            record["top_document_correct"] = bool(
                retrieved_documents and retrieved_documents[0] in question.relevant_document_ids
            )
        per_question.append(record)

    answerable_records = [r for r in per_question if r["answerable"]]

    aggregate = {}
    for k in K_VALUES:
        aggregate[f"hit_rate@{k}"] = mean(r["metrics"][f"hit_rate@{k}"] for r in answerable_records)
        aggregate[f"recall@{k}"] = mean(r["metrics"][f"recall@{k}"] for r in answerable_records)
        aggregate[f"precision@{k}"] = mean(
            r["metrics"][f"precision@{k}"] for r in answerable_records
        )
    aggregate["mrr"] = mean(r["metrics"]["reciprocal_rank"] for r in answerable_records)
    aggregate["top_document_accuracy"] = mean(
        float(r["top_document_correct"]) for r in answerable_records
    )

    by_category: dict[str, dict] = {}
    for record in answerable_records:
        bucket = by_category.setdefault(record["category"], {"n": 0, "hit_rate@5": [], "mrr": []})
        bucket["n"] += 1
        bucket["hit_rate@5"].append(record["metrics"]["hit_rate@5"])
        bucket["mrr"].append(record["metrics"]["reciprocal_rank"])
    by_category = {
        name: {
            "n": bucket["n"],
            "hit_rate@5": round(mean(bucket["hit_rate@5"]), 4),
            "mrr": round(mean(bucket["mrr"]), 4),
        }
        for name, bucket in sorted(by_category.items())
    }

    by_form: dict[str, dict] = {}
    for record in answerable_records:
        for document_id in record["relevant_document_ids"]:
            bucket = by_form.setdefault(document_id, {"n": 0, "hit_rate@5": [], "mrr": []})
            bucket["n"] += 1
            bucket["hit_rate@5"].append(record["metrics"]["hit_rate@5"])
            bucket["mrr"].append(record["metrics"]["reciprocal_rank"])
    by_form = {
        name: {
            "n": bucket["n"],
            "hit_rate@5": round(mean(bucket["hit_rate@5"]), 4),
            "mrr": round(mean(bucket["mrr"]), 4),
        }
        for name, bucket in sorted(by_form.items())
    }

    unanswerable_records = [r for r in per_question if not r["answerable"]]
    unanswerable_scores = [r["top_score"] for r in unanswerable_records if r["top_score"]]
    answerable_scores = [r["top_score"] for r in answerable_records if r["top_score"]]

    ordered_latencies = sorted(latencies)
    result = {
        "run": {
            "benchmark": "nfip-sfip-retrieval",
            "split": args.split,
            "retriever": "dense",
            "note": (
                "Dense BGE retrieval only. No lexical baseline, no reranking, no "
                "generation evaluation. Unanswerable questions are excluded from every "
                "aggregate below and reported separately."
            ),
        },
        "configuration": {
            "embedding_model": cfg.EMBEDDING_MODEL,
            "embedding_dimension": 384,
            "query_prefix": cfg.BGE_QUERY_PREFIX,
            "distance": "cosine",
            "chunk_size": cfg.CHUNK_SIZE,
            "chunk_overlap": cfg.CHUNK_OVERLAP,
            "top_k": args.top_k,
            "collection": cfg.QDRANT_COLLECTION,
            "config_hash": config_hash(
                embedding_model=cfg.EMBEDDING_MODEL,
                size=cfg.CHUNK_SIZE,
                overlap=cfg.CHUNK_OVERLAP,
            ),
        },
        "corpus": {
            "documents": len(documents),
            "chunks": len(chunks),
            "characters": sum(d.characters for d in documents),
            "words": sum(len(d.text.split()) for d in documents),
            "corpus_hash": corpus_hash(documents),
            "per_document": {
                d.document_id: {
                    "chunks": sum(1 for c in chunks if c.document_id == d.document_id),
                    "characters": d.characters,
                    "sha256": d.sha256,
                }
                for d in documents
            },
        },
        "questions": {
            **summarise(questions),
            "question_set_hash": question_set_hash(all_questions),
            "evaluated": len(questions),
        },
        "retrieval_metrics_answerable_only": {
            key: round(value, 4) for key, value in aggregate.items()
        },
        "by_category": by_category,
        "by_relevant_form": by_form,
        "unanswerable": {
            "count": len(unanswerable_records),
            "note": (
                "No recall is computed: these questions have no relevant chunk. Top-1 "
                "similarity is descriptive only. No abstention threshold is proposed - "
                "that needs a threshold study on the dev split, which has not been run."
            ),
            "top_score_min": round(min(unanswerable_scores), 4) if unanswerable_scores else None,
            "top_score_max": round(max(unanswerable_scores), 4) if unanswerable_scores else None,
            "top_score_mean": round(mean(unanswerable_scores), 4) if unanswerable_scores else None,
            "answerable_top_score_mean": (
                round(mean(answerable_scores), 4) if answerable_scores else None
            ),
        },
        "latency_ms": {
            "n": len(latencies),
            "mean": round(mean(latencies), 2),
            "p50": round(ordered_latencies[len(ordered_latencies) // 2], 2),
            "p95": round(ordered_latencies[int(len(ordered_latencies) * 0.95)], 2),
            "max": round(max(latencies), 2),
            "note": "End-to-end query embedding plus vector search, local Qdrant.",
        },
        "per_question": per_question,
    }

    Path(args.out).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"split={args.split}  questions={len(questions)} "
          f"(answerable {len(answerable)}, unanswerable {len(unanswerable)})")
    for key, value in result["retrieval_metrics_answerable_only"].items():
        print(f"  {key:24} {value}")
    print(f"  latency p50/p95 ms       {result['latency_ms']['p50']} / {result['latency_ms']['p95']}")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
