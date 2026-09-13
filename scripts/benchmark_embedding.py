"""Time the embedding step, which is the only part of this system a GPU changes.

Retrieval here is BM25 plus a dense index. BM25 is pure Python over a few hundred chunks
and generation is a separate Ollama process, so the embedder is the one component whose
device matters: it runs over the whole corpus at ingest and once per query at serving time.

Two measurements, because they are different workloads:

  corpus  - 314 chunks in one batch, what ingestion does
  query   - a single short string, what a request does, where fixed overheads dominate

The accelerator's vectors are compared against the CPU's. Cosine similarity is the number
that matters: these embeddings are normalised and searched by cosine distance, so a device
that shifts them changes which chunk is retrieved, and a speed-up bought that way is not a
speed-up. Ingest and serve on the same device, or re-ingest when you change it.

    python scripts/benchmark_embedding.py [--json out.json]

Not wired into CI, which has no GPU and where a timing would mean nothing anyway.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.config import cfg
from src.corpus import chunk_corpus, load_corpus

QUERY = "What is the Increased Cost of Compliance limit?"


def available_devices() -> list[str]:
    import torch

    devices = ["cpu"]
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devices.append("xpu")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def device_name(device: str) -> str:
    import torch

    if device == "xpu":
        return torch.xpu.get_device_properties(0).name
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    return "CPU"


def synchronize(device: str) -> None:
    import torch

    if device == "xpu":
        torch.xpu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def time_encode(model, texts, device: str, repeats: int, warmup: int):
    for _ in range(warmup):
        vectors = model.encode(texts, normalize_embeddings=True)
    synchronize(device)
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        vectors = model.encode(texts, normalize_embeddings=True)
        synchronize(device)
        timings.append((time.perf_counter() - started) * 1000)
    return timings, vectors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--query-repeats", type=int, default=50)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    import numpy as np
    from sentence_transformers import SentenceTransformer

    chunks = chunk_corpus(load_corpus(REPO / "data" / "corpus"), cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
    texts = [chunk.text for chunk in chunks]
    print(f"model   {cfg.EMBEDDING_MODEL}")
    print(f"corpus  {len(texts)} chunks at {cfg.CHUNK_SIZE}/{cfg.CHUNK_OVERLAP}")
    print(f"repeats {args.repeats} corpus, {args.query_repeats} query, {args.warmup} warm-up\n")

    record: dict[str, dict] = {
        "model": cfg.EMBEDDING_MODEL,
        "chunks": len(texts),
        "chunking": {"size": cfg.CHUNK_SIZE, "overlap": cfg.CHUNK_OVERLAP},
        "devices": {},
    }
    vectors_by_device: dict[str, object] = {}

    print(f"  {'device':6s} {'hardware':34s} {'corpus':>11s} {'per chunk':>10s} {'query':>9s}")
    for device in available_devices():
        model = SentenceTransformer(cfg.EMBEDDING_MODEL, device=device)
        corpus_times, vectors = time_encode(model, texts, device, args.repeats, args.warmup)
        query_times, _ = time_encode(model, QUERY, device, args.query_repeats, args.warmup)
        vectors_by_device[device] = np.asarray(vectors)

        corpus_median = statistics.median(corpus_times)
        query_median = statistics.median(query_times)
        record["devices"][device] = {
            "hardware": device_name(device),
            "corpus_median_ms": round(corpus_median, 1),
            "corpus_min_ms": round(min(corpus_times), 1),
            "per_chunk_ms": round(corpus_median / len(texts), 3),
            "query_median_ms": round(query_median, 2),
            "query_p95_ms": round(sorted(query_times)[int(0.95 * len(query_times)) - 1], 2),
        }
        print(f"  {device:6s} {device_name(device)[:34]:34s} {corpus_median:9.0f}ms "
              f"{corpus_median / len(texts):9.2f}ms {query_median:8.2f}ms")

    baseline = vectors_by_device["cpu"]
    for device, vectors in vectors_by_device.items():
        if device == "cpu":
            continue
        # Row-wise cosine similarity. The vectors are already unit length.
        cosine = float((baseline * vectors).sum(axis=1).min())
        gap = float(np.abs(baseline - vectors).max())
        record["devices"][device]["min_cosine_vs_cpu"] = round(cosine, 8)
        record["devices"][device]["max_abs_diff_vs_cpu"] = float(f"{gap:.3e}")
        speedup = (record["cpu"]["corpus_median_ms"] if "cpu" in record else
                   record["devices"]["cpu"]["corpus_median_ms"]) / record["devices"][device]["corpus_median_ms"]
        print(f"\n  {device} vs cpu: corpus {speedup:.2f}x, "
              f"query {record['devices']['cpu']['query_median_ms'] / record['devices'][device]['query_median_ms']:.2f}x")
        print(f"  minimum cosine similarity between the two sets of vectors: {cosine:.8f}")
        if cosine < 0.9999:
            print("  WARNING: the vectors differ enough to change retrieval. "
                  "Ingest and serve on the same device.")

    if args.json:
        args.json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
