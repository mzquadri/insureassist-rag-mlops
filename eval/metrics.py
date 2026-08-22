"""
Retrieval metrics.

Every function here is pure and deterministic: ranked IDs and a relevance set in, a number
out. No model, no database, no configuration. That is what lets them be tested against
hand-computed answers rather than against whatever the system happens to produce.

Two defensive decisions worth stating, because both change results silently if got wrong:

  * **Duplicate retrieved IDs are collapsed, keeping the first occurrence.** A retriever
    returning the same chunk twice must not thereby score two hits, and must not have its
    ranks shifted for everything below.
  * **An empty relevance set returns NaN, not zero.** Zero is a real score meaning "found
    nothing relevant". A question with no relevant chunks - an unanswerable one - has no
    recall to measure, and averaging a fabricated zero into the mean would drag it down
    while looking like a genuine failure.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def deduplicate(retrieved: Iterable[str]) -> list[str]:
    """Drop repeats, keeping first position."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in retrieved:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def hit_rate_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if any relevant item appears in the top k, else 0.0.

    Answers "did the retriever surface the evidence at all", which for a question with a
    single correct chunk is the measure that matters most.
    """
    if not relevant:
        return float("nan")
    top = deduplicate(retrieved)[:k]
    return 1.0 if any(item in relevant for item in top) else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant items that appear in the top k.

    Differs from hit rate only when a question has more than one relevant chunk - which is
    exactly what the multi_chunk questions are for.
    """
    if not relevant:
        return float("nan")
    top = deduplicate(retrieved)[:k]
    return len([item for item in top if item in relevant]) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top k that is relevant.

    Bounded above by len(relevant)/k, so with one relevant chunk and k=5 the ceiling is
    0.2. It is reported for completeness, not as a headline number.
    """
    if not relevant:
        return float("nan")
    top = deduplicate(retrieved)[:k]
    if not top:
        return 0.0
    return len([item for item in top if item in relevant]) / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant item, or 0.0 if none was retrieved."""
    if not relevant:
        return float("nan")
    for position, item in enumerate(deduplicate(retrieved), start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def mean_reciprocal_rank(results: Iterable[tuple[Sequence[str], set[str]]]) -> float:
    scores = [reciprocal_rank(r, rel) for r, rel in results]
    usable = [s for s in scores if not math.isnan(s)]
    return sum(usable) / len(usable) if usable else float("nan")


def mean(values: Iterable[float]) -> float:
    """Mean that ignores NaN, so unanswerable questions cannot distort an average."""
    usable = [v for v in values if not math.isnan(v)]
    return sum(usable) / len(usable) if usable else float("nan")


# ---------------------------------------------------------------------------------------
# nDCG is deliberately NOT implemented.
#
# nDCG exists to reward putting *more* relevant results above *less* relevant ones, which
# needs graded relevance. Every label in this benchmark is binary: a chunk either contains
# the provision or it does not, and there is no basis in the source text for saying one
# chunk is twice as relevant as another.
#
# With binary labels and usually a single relevant chunk, nDCG@k is a monotone function of
# the same reciprocal rank already reported - it would add a column, not information, while
# implying a grading effort that was never made. It is deferred until labels justify it.
# ---------------------------------------------------------------------------------------
