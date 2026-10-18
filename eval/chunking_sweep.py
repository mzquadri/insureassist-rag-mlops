"""
Re-derive the dense chunking sweep on the DEV split, optionally tracked in MLflow.

    python -m eval.chunking_sweep                         # writes the JSON only
    python -m eval.chunking_sweep --mlflow                # also logs every config to MLflow
    mlflow ui                                             # browse the runs

Why this exists: `eval/dev_chunking_sweep.json` is the evidence for choosing 800/120, but no
script in the repository produces it, so that choice could not be re-run. This script makes
the selection step reproducible. It does not overwrite the original file; it writes
`eval/dev_chunking_sweep_reproduced.json` so the two can be compared.

What it measures, exactly:
  * DEV split, answerable questions only. TEST is never read.
  * Dense retrieval only, with the frozen embedder (cfg.EMBEDDING_MODEL) and the BGE query
    prefix. Search is exact cosine over normalised vectors in memory, not Qdrant, so the
    sweep needs no running database. With a few hundred chunks Qdrant also searches
    exhaustively, so the ranking is the same up to ties.
  * Relevance at each chunking is re-derived from the hand-authored evidence spans: a chunk
    is relevant when it fully contains an evidence span. At 800/120 this rule reproduces the
    stored `relevant_chunk_ids` exactly, and the script refuses to run if it ever does not.
  * A question none of whose spans fits inside a single chunk at some setting cannot be
    scored there. It is counted in `unscorable` rather than silently dropped.

MLflow is optional and imported only with --mlflow (`pip install -r requirements-mlops.txt`).
One parent run holds the sweep; each chunk configuration is a nested child run with its
parameters, metrics and dataset hashes.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from eval.ground_truth import Question, load_questions, question_set_hash
from eval.metrics import hit_rate_at_k, mean, recall_at_k, reciprocal_rank
from src.config import cfg
from src.corpus import Chunk, chunk_corpus, corpus_hash, load_corpus
from src.providers import get_embedder

#: The grid recorded in eval/retrieval_config.json ("Selected on dev from ...").
DEFAULT_CONFIGS = ((400, 80), (600, 100), (600, 150), (800, 120), (1000, 150))
#: The frozen configuration, at which derived labels must equal the stored ones.
FROZEN = (800, 120)
RETRIEVAL_DEPTH = 10
DEFAULT_OUT = "eval/dev_chunking_sweep_reproduced.json"


def relevant_chunks(question: Question, chunks: list[Chunk]) -> set[str]:
    """Chunks that fully contain at least one of the question's evidence spans."""
    return {
        c.chunk_id
        for c in chunks
        for s in question.evidence_spans
        if s.document_id == c.document_id and c.start <= s.start and s.end <= c.end
    }


def check_labels_at_frozen(questions: list[Question], documents) -> None:
    """Fail loudly if the relevance rule disagrees with the stored labels at 800/120."""
    chunks = chunk_corpus(documents, *FROZEN)
    mismatched = [
        q.question_id
        for q in questions
        if q.answerable and relevant_chunks(q, chunks) != set(q.relevant_chunk_ids)
    ]
    if mismatched:
        raise SystemExit(
            f"relevance rule disagrees with stored labels at {FROZEN[0]}/{FROZEN[1]} for "
            f"{mismatched}; the sweep would not be comparable with the benchmark"
        )


def embed(texts: list[str], prefix: str = "") -> np.ndarray:
    vectors = get_embedder().encode([prefix + t for t in texts], normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


def score_config(questions: list[Question], documents, size: int, overlap: int) -> dict:
    chunks = chunk_corpus(documents, size, overlap)
    chunk_ids = [c.chunk_id for c in chunks]
    started = time.perf_counter()
    matrix = embed([c.text for c in chunks])
    embed_seconds = time.perf_counter() - started

    rows, unscorable = [], []
    for question in questions:
        relevant = relevant_chunks(question, chunks)
        if not relevant:
            unscorable.append(question.question_id)
            continue
        query = embed([question.question], prefix=cfg.BGE_QUERY_PREFIX)[0]
        order = np.argsort(-(matrix @ query), kind="stable")[:RETRIEVAL_DEPTH]
        ids = [chunk_ids[i] for i in order]
        top_document = ids[0].split("#")[0]
        rows.append({
            "hit1": hit_rate_at_k(ids, relevant, 1),
            "hit3": hit_rate_at_k(ids, relevant, 3),
            "hit5": hit_rate_at_k(ids, relevant, 5),
            "rec5": recall_at_k(ids, relevant, 5),
            "mrr": reciprocal_rank(ids, relevant),
            "topdoc": float(top_document in question.relevant_document_ids),
        })

    metrics = {
        name: round(mean(r[name] for r in rows), 4) if rows else math.nan
        for name in ("hit1", "hit3", "hit5", "rec5", "mrr", "topdoc")
    }
    return {
        "size": size,
        "overlap": overlap,
        "chunks": len(chunks),
        "n": len(rows),
        "unscorable": unscorable,
        "embed_seconds": round(embed_seconds, 2),
        **metrics,
    }


def _parse_configs(text: str) -> tuple[tuple[int, int], ...]:
    pairs = []
    for item in text.split(","):
        size, overlap = item.strip().split("/")
        pairs.append((int(size), int(overlap)))
    return tuple(pairs)


def run(configs, out: Path, use_mlflow: bool, experiment: str) -> dict:
    documents = load_corpus()
    all_questions = load_questions()
    check_labels_at_frozen(all_questions, documents)
    dev = [q for q in all_questions if q.split == "dev" and q.answerable]

    stamp = {
        "split": "dev",
        "answerable_questions": len(dev),
        "embedding_model": cfg.EMBEDDING_MODEL,
        "retrieval": f"dense, exact cosine, depth {RETRIEVAL_DEPTH}",
        "relevance": "chunk fully contains an evidence span",
        "corpus_hash": corpus_hash(documents),
        "question_set_hash": question_set_hash(all_questions),
    }
    results = {f"{s}/{o}": score_config(dev, documents, s, o) for s, o in configs}
    report = {"stamp": stamp, "configs": results}

    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    for key, r in results.items():
        print(f"  {key:>9}  chunks {r['chunks']:>4}  n {r['n']:>2}  hit@5 {r['hit5']:.3f}  "
              f"mrr {r['mrr']:.3f}  topdoc {r['topdoc']:.3f}")
    print(f"\nWrote {out}")

    if use_mlflow:
        log_to_mlflow(report, out, experiment)
    return report


def log_to_mlflow(report: dict, out: Path, experiment: str) -> None:
    import mlflow  # optional dependency, see requirements-mlops.txt

    stamp = report["stamp"]
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="dev-chunking-sweep"):
        mlflow.set_tags({"split": "dev", "stage": "selection"})
        mlflow.log_params({k: v for k, v in stamp.items() if k != "answerable_questions"})
        mlflow.log_param("answerable_questions", stamp["answerable_questions"])
        mlflow.log_artifact(str(out))
        for key, r in report["configs"].items():
            with mlflow.start_run(run_name=f"chunk-{key}", nested=True):
                mlflow.log_params({
                    "chunk_size": r["size"],
                    "chunk_overlap": r["overlap"],
                    "embedding_model": stamp["embedding_model"],
                })
                mlflow.log_metrics({
                    "chunks": r["chunks"],
                    "questions_scored": r["n"],
                    "questions_unscorable": len(r["unscorable"]),
                    "embed_seconds": r["embed_seconds"],
                    **{
                        name: r[name]
                        for name in ("hit1", "hit3", "hit5", "rec5", "mrr", "topdoc")
                        if not math.isnan(r[name])
                    },
                })
    print(f"Logged {len(report['configs'])} configurations to MLflow experiment "
          f"'{experiment}' (tracking URI: {mlflow.get_tracking_uri()})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", default=",".join(f"{s}/{o}" for s, o in DEFAULT_CONFIGS),
                        help="comma-separated size/overlap pairs, e.g. 600/100,800/120")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--mlflow", action="store_true", help="log the sweep to MLflow")
    parser.add_argument("--experiment", default="insureassist-chunking")
    args = parser.parse_args()
    run(_parse_configs(args.configs), Path(args.out), args.mlflow, args.experiment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
