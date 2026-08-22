"""
The canonical reference run.

    python -m eval.reference_run --out eval/reference_run.json

One command, one machine-readable artefact. Everything published about this project's
performance is derived from this file; the README is never the source of truth.

Covers, in order:
  * retrieval on the held-out TEST split, for the selected architecture and both baselines
  * deterministic citation metrics
  * abstention behaviour
  * optional generation, only when a generator is reachable

Judge-based answer scoring is deliberately absent. Only one local model is available, and
using it to grade its own output is the circular evaluation this project already removed
once. See docs/EVALUATION.md.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
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
from src.providers import get_bm25_index
from src.rag import CANDIDATE_DEPTH, citation, retrieve
from src.retrieval import (
    load_retrieval_config,
    reciprocal_rank_fusion,
    retrieval_config_hash,
)

SCHEMA_VERSION = 2
K_VALUES = (1, 3, 5)


def _dense_only(question: str, limit: int) -> list[str]:
    from src.rag import _dense_candidates

    hits, _ = _dense_candidates(question, limit)
    return [h.chunk_id for h in hits][:limit]


def _bm25_only(question: str, limit: int) -> list[str]:
    return [h.chunk_id for h in get_bm25_index().search(question, limit)]


def _hybrid(question: str, limit: int) -> list[str]:
    from src.rag import _dense_candidates

    dense, _ = _dense_candidates(question, CANDIDATE_DEPTH)
    lexical = get_bm25_index().search(question, CANDIDATE_DEPTH)
    return [h.chunk_id for h in reciprocal_rank_fusion([dense, lexical], limit=limit)]


def score_retriever(questions, search, depth: int) -> dict:
    """Retrieval metrics over the answerable questions only."""
    rows = []
    for question in questions:
        if not question.answerable:
            continue
        ids = search(question.question, depth)
        relevant = set(question.relevant_chunk_ids)

        documents = []
        for chunk_id in ids:
            document_id = chunk_id.split("#")[0]
            if document_id not in documents:
                documents.append(document_id)

        rows.append({
            "question_id": question.question_id,
            "category": question.category,
            "relevant_document_ids": question.relevant_document_ids,
            "retrieved_chunk_ids": ids,
            "retrieved_document_ids": documents,
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

    return {
        "metrics": metrics,
        "by_category": {
            name: {
                "n": b["n"],
                "hit_rate@5": round(mean(b["hit"]), 4),
                "mrr": round(mean(b["mrr"]), 4),
            }
            for name, b in sorted(by_category.items())
        },
        "per_question": rows,
    }


def citation_metrics(questions, documents) -> dict:
    """Deterministic: do the citations returned by the API point at the right text?

    No model involved. Citation precision is the share of returned citations that are
    labelled relevant; recall is the share of labelled relevant chunks that were cited.
    `unsupported_citation_rate` counts citations whose offsets do not reproduce the cited
    text from the committed corpus - it must be zero, and a non-zero value means the
    service is fabricating provenance.
    """
    by_id = {d.document_id: d for d in documents}
    precisions, recalls, unsupported, total = [], [], 0, 0

    for question in questions:
        if not question.answerable:
            continue
        contexts = retrieve(question.question, cfg.TOP_K)
        citations = [citation(c) for c in contexts]
        relevant = set(question.relevant_chunk_ids)
        cited = [c["chunk_id"] for c in citations]

        for entry in citations:
            total += 1
            document = by_id.get(entry["document_id"])
            source_text = document.text[entry["start"]:entry["end"]] if document else ""
            excerpt = entry["excerpt"].removesuffix("...")
            if not source_text.startswith(excerpt.rstrip()):
                unsupported += 1

        if cited:
            precisions.append(len([c for c in cited if c in relevant]) / len(cited))
        recalls.append(len([c for c in cited if c in relevant]) / len(relevant))

    return {
        "citation_precision": round(mean(precisions), 4),
        "citation_recall": round(mean(recalls), 4),
        "unsupported_citation_rate": round(unsupported / total, 4) if total else 0.0,
        "citations_checked": total,
        "note": (
            "Deterministic, no model. Precision is bounded by the number of relevant "
            "chunks divided by top_k, so it cannot approach 1.0 when a question has one "
            "relevant chunk and five are returned."
        ),
    }


def abstention_metrics(questions) -> dict:
    """Does the service abstain when it should, and answer when it should?"""
    answerable_answered = unanswerable_answered = 0
    answerable_total = unanswerable_total = 0

    for question in questions:
        contexts = retrieve(question.question, cfg.TOP_K)
        answered = bool(contexts)
        if question.answerable:
            answerable_total += 1
            answerable_answered += answered
        else:
            unanswerable_total += 1
            unanswerable_answered += answered

    return {
        "policy": "structural only: abstain when retrieval returns no chunk",
        "answerable_acceptance_rate": (
            round(answerable_answered / answerable_total, 4) if answerable_total else None
        ),
        "unanswerable_rejection_rate": (
            round(1 - unanswerable_answered / unanswerable_total, 4)
            if unanswerable_total else None
        ),
        "false_abstentions": answerable_total - answerable_answered,
        "false_answers": unanswerable_answered,
        "note": (
            "No similarity threshold is applied. On the dev split no single-threshold rule "
            "on top-1 dense score beat the always-answer baseline (0.778 vs 0.778), and the "
            "hardest unanswerable question scored above the answerable mean. With 8 "
            "unanswerable questions in total the benchmark is too small to support a "
            "threshold claim, so none is made. Every unanswerable question therefore "
            "retrieves something and is passed to the generator, which is a real weakness "
            "and is recorded as one."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="eval/reference_run.json")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--with-generation", action="store_true",
                        help="also run the generator (needs Ollama); off by default")
    args = parser.parse_args()

    config = load_retrieval_config()
    documents = load_corpus()
    chunks = chunk_corpus(documents, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
    all_questions = load_questions()
    questions = [q for q in all_questions if args.split == "all" or q.split == args.split]

    depth = config["retrieval_depth"]

    started = time.perf_counter()
    selected = score_retriever(questions, _hybrid, depth)
    hybrid_seconds = time.perf_counter() - started

    baselines = {
        "dense": score_retriever(questions, _dense_only, depth),
        "bm25": score_retriever(questions, _bm25_only, depth),
    }

    latencies = []
    for question in questions:
        t0 = time.perf_counter()
        retrieve(question.question, cfg.TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)
    ordered = sorted(latencies)

    result = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "nfip-sfip-retrieval",
        "split": args.split,
        "corpus": {
            "documents": len(documents),
            "chunks": len(chunks),
            "characters": sum(d.characters for d in documents),
            "words": sum(len(d.text.split()) for d in documents),
            "corpus_hash": corpus_hash(documents),
            "document_hashes": {d.document_id: d.sha256 for d in documents},
        },
        "questions": {
            **summarise(all_questions),
            "evaluated_in_this_run": len(questions),
            "question_set_hash": question_set_hash(all_questions),
        },
        "retrieval": {
            "architecture": config["architecture"],
            "selected_on": "dev split",
            "dense_model": config["dense"]["model"],
            "bm25": {k: config["bm25"][k] for k in ("k1", "b", "tokenizer", "stopwords")},
            "fusion": config["fusion"],
            "chunking": config["chunking"],
            "retrieval_depth": depth,
            "serving_top_k": config["serving_top_k"],
            "metrics": selected["metrics"],
            "by_category": selected["by_category"],
            "baselines": {name: b["metrics"] for name, b in baselines.items()},
            "baseline_by_category": {name: b["by_category"] for name, b in baselines.items()},
        },
        "generation": {
            "generator": cfg.OLLAMA_MODEL if cfg.LLM_BACKEND == "ollama" else cfg.HF_BASE_MODEL,
            "backend": cfg.LLM_BACKEND,
            "fine_tuned": False,
            "citations": citation_metrics(questions, documents),
            "abstention": abstention_metrics(questions),
            "judge_metrics": None,
            "judge_note": (
                "No judge metrics are published. Only one local model is available, and "
                "using it to grade its own output is circular. A judge would have to be a "
                "different model; none is installed, and downloading one purely to fill "
                "this field would not make the number more trustworthy."
            ),
        },
        "latency_ms": {
            "n": len(latencies),
            "mean": round(mean(latencies), 2),
            "p50": round(ordered[len(ordered) // 2], 2),
            "p95": round(ordered[int(len(ordered) * 0.95)], 2),
            "note": "Machine-specific. Not a property of the system.",
        },
        "reproducibility": {
            "config_hash": config_hash(
                embedding_model=cfg.EMBEDDING_MODEL,
                size=cfg.CHUNK_SIZE,
                overlap=cfg.CHUNK_OVERLAP,
            ),
            "retrieval_config_hash": retrieval_config_hash(config),
            "python": platform.python_version(),
            "seed": "not applicable: retrieval is deterministic; generation is not scored here",
            "hybrid_eval_seconds": round(hybrid_seconds, 2),
        },
        "per_question": selected["per_question"],
    }

    Path(args.out).write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"split={args.split}  questions={len(questions)}")
    print(f"  architecture: {config['architecture']}")
    for name, value in selected["metrics"].items():
        print(f"    {name:24} {value}")
    print("  baselines:")
    for name, baseline in baselines.items():
        m = baseline["metrics"]
        print(f"    {name:8} hit@5 {m['hit_rate@5']:.3f}  mrr {m['mrr']:.3f}  "
              f"topdoc {m['top_document_accuracy']:.3f}")
    citations = result["generation"]["citations"]
    print(f"  citation precision {citations['citation_precision']} / "
          f"recall {citations['citation_recall']} / "
          f"unsupported {citations['unsupported_citation_rate']}")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
