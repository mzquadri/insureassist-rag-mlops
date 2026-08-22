"""
Retrievers over the NFIP chunks: lexical (BM25), dense (BGE), and a fusion of both.

All three return `(chunk_id, score)` pairs over the *same* chunks, with the same IDs and
metadata, so their results are directly comparable and can be scored by the same metrics.

BM25 is implemented here rather than pulled from a library. It is about eighty lines, it
removes a dependency, and - more to the point - it makes the tokenizer, the parameters and
the scoring formula visible and testable instead of hidden behind a default. A baseline
whose behaviour you cannot inspect is not much of a baseline.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from src.corpus import Chunk

# ---------------------------------------------------------------------------------------
# Tokenisation
#
# Deliberately minimal and deterministic:
#   * lowercase
#   * split on any run of non-alphanumeric characters, keeping digits attached to digits
#   * no stemming, no lemmatisation, no synonym or query expansion
#   * no stop-word list
#
# No stop-words because this corpus is legal prose where "not", "other than", "unless" and
# "no" carry the meaning - an exclusion clause is mostly stop-words. Removing them would
# damage exactly the questions this benchmark is built around.
#
# Monetary amounts survive intact: "$30,000" becomes "30" and "000" under a naive split, so
# digit groups separated by commas are joined first. "30,000" -> "30000".
# ---------------------------------------------------------------------------------------
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, join thousands separators, split on non-alphanumeric runs."""
    return _TOKEN.findall(_THOUSANDS.sub("", text.lower()))


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float


class BM25Index:
    """Okapi BM25 over the corpus chunks.

    Defaults are the standard k1=1.5, b=0.75. They were not tuned; see docs/BENCHMARK.md
    for the dev-set check that confirmed tuning them was not worth it here.
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunk_ids = [c.chunk_id for c in chunks]

        self.documents = [tokenize(c.text) for c in chunks]
        self.lengths = [len(d) for d in self.documents]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0

        self.term_frequencies = [Counter(d) for d in self.documents]

        document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequency.update(frequencies.keys())

        # Robertson/Sparck-Jones idf with the +0.5 smoothing, floored at a small positive
        # value. Without the floor a term appearing in more than half the chunks gets a
        # negative idf and actively penalises documents that contain it.
        total = len(self.documents)
        self.idf = {
            term: max(
                math.log((total - count + 0.5) / (count + 0.5) + 1.0),
                1e-9,
            )
            for term, count in document_frequency.items()
        }

    def search(self, query: str, limit: int) -> list[Hit]:
        query_terms = tokenize(query)
        scores: list[float] = [0.0] * len(self.documents)

        for term in query_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, frequencies in enumerate(self.term_frequencies):
                frequency = frequencies.get(term)
                if not frequency:
                    continue
                normalisation = 1 - self.b + self.b * (
                    self.lengths[index] / self.average_length if self.average_length else 0
                )
                scores[index] += idf * (
                    frequency * (self.k1 + 1) / (frequency + self.k1 * normalisation)
                )

        # Sort by score, then by chunk_id, so ties resolve identically on every run.
        ranked = sorted(
            ((score, self.chunk_ids[i]) for i, score in enumerate(scores) if score > 0),
            key=lambda pair: (-pair[0], pair[1]),
        )
        return [Hit(chunk_id=chunk_id, score=score) for score, chunk_id in ranked[:limit]]


def reciprocal_rank_fusion(
    rankings: list[list[Hit]], k: int = 60, limit: int = 10
) -> list[Hit]:
    """Combine ranked lists by rank position rather than by score.

    Dense cosine similarity and BM25 scores are on different, unnormalised scales, so they
    cannot be added or averaged without inventing a weighting. RRF only reads positions:
    each list contributes 1/(k + rank), which needs no calibration between the two systems
    and no per-corpus tuning. k=60 is the value from the original Cormack et al. paper and
    is left alone.
    """
    totals: dict[str, float] = {}
    for ranking in rankings:
        for position, hit in enumerate(ranking, start=1):
            totals[hit.chunk_id] = totals.get(hit.chunk_id, 0.0) + 1.0 / (k + position)

    ranked = sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))
    return [Hit(chunk_id=chunk_id, score=score) for chunk_id, score in ranked[:limit]]


# ---------------------------------------------------------------------------------------
# The frozen retrieval configuration.
#
# Committed as JSON rather than left in code so a future change is a visible diff to an
# artefact, not an edit buried in a module - and so the reference run can hash it.
# ---------------------------------------------------------------------------------------
import json
from pathlib import Path

RETRIEVAL_CONFIG_PATH = Path("eval/retrieval_config.json")


def load_retrieval_config(path: Path = RETRIEVAL_CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def retrieval_config_hash(config: dict) -> str:
    """Hash of the settings that determine retrieval behaviour."""
    import hashlib

    material = json.dumps(
        {
            "architecture": config["architecture"],
            "dense": config["dense"],
            "bm25": {k: v for k, v in config["bm25"].items() if k in ("k1", "b", "tokenizer")},
            "fusion": {k: v for k, v in config["fusion"].items() if k in ("method", "k", "candidate_depth_per_retriever")},
            "chunking": {k: v for k, v in config["chunking"].items() if k in ("size", "overlap")},
            "retrieval_depth": config["retrieval_depth"],
        },
        sort_keys=True,
    )
    return hashlib.sha256(material.encode()).hexdigest()
