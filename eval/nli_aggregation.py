"""Using a sentence-pair NLI model on premises longer than a sentence pair.

Two judges failed the control in `eval/citation_correctness_run.py`, in opposite
directions, and both failures trace to the same two mistakes.

**The premise was too long.** Chunks are cut at 800 characters, about 130 words, where
ALCE segments its corpora into 100-word passages and the NLI models were trained on
sentence pairs. The published answer to a premise outside that distribution is to split
it, score each sentence against the hypothesis, and aggregate: "retrieve-and-classify",
Stretching Sentence-pair NLI Models to Reason over Long Documents (arXiv:2204.07447).
This moves the premise *towards* ALCE's granularity rather than away from it.

**The decision was an argmax.** NLI models emit probabilities that need a per-task cut;
taking the argmax applies whatever cut the training mix happened to leave behind. The
threshold here is chosen on the control by Youden's J, which is the method
`eval/abstention_threshold.py` already uses in this repository for the same kind of
decision.

Both functions take the scorer as an argument, so neither needs a model to be tested.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

#: Sentence split for premises. Deliberately the same shape as the one in
#: citation_correctness, and deliberately simple: a premise cut into slightly wrong
#: pieces still scores better than one the model was never built to read.
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

Scorer = Callable[[str, str], float]


def premise_parts(premise: str) -> list[str]:
    """Each sentence, plus the undivided premise.

    The whole is kept in the pool because some claims are supported only by two clauses
    together and no single sentence entails them. Dropping it would trade one blind spot
    for another.
    """
    text = (premise or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE.split(text) if p.strip()]
    if len(parts) > 1:
        parts.append(text)
    return parts


def best_sentence_score(premise: str, hypothesis: str, score: Scorer) -> float:
    """The strongest support any single part of the premise offers.

    Max rather than mean. Support from one clause is support, and averaging lets a long
    block of unrelated policy text bury the sentence that actually says it.
    """
    parts = premise_parts(premise)
    if not parts:
        return 0.0
    return max(score(part, hypothesis) for part in parts)


def best_scores(
    pairs: Sequence[tuple[str, str]], score_many: Callable[[list], list[float]]
) -> list[float]:
    """`best_sentence_score` for many pairs, in one pass.

    Splitting the premise multiplies the work by the number of sentences in it, and a
    run that issues one forward pass per sentence does not finish: roughly 1,300 of them
    for a single judge here. Every (part, hypothesis) across every pair is flattened into
    one list, scored together, and regrouped by taking the max within each pair.
    """
    flat: list[tuple[str, str]] = []
    spans: list[tuple[int, int]] = []
    for premise, hypothesis in pairs:
        parts = premise_parts(premise)
        start = len(flat)
        flat.extend((part, hypothesis) for part in parts)
        spans.append((start, len(flat)))

    scored = score_many(flat) if flat else []
    return [max(scored[a:b], default=0.0) for a, b in spans]


def batch_plan(lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Group pair indices into batches costing at most `budget` padded tokens.

    A fixed batch count does not survive on a small card. Cost is batch size times the
    *padded* length, and padding is set by the longest member, so one 512-token premise
    makes a batch of 32 cost what 32 long premises would. Sorting by length first means
    the sentence-level pairs - most of them, and short - batch widely, while the few
    whole-premise pairs batch narrowly or alone.

    A pair longer than the whole budget is still returned, in a batch of its own: the
    model truncates it, and silently dropping a pair would bias the rate it feeds.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: list[list[int]] = []
    current: list[int] = []
    widest = 0
    for index in order:
        candidate_width = max(widest, lengths[index])
        if current and candidate_width * (len(current) + 1) > budget:
            batches.append(current)
            current, widest = [index], lengths[index]
        else:
            current.append(index)
            widest = candidate_width
    if current:
        batches.append(current)
    return batches


def youden_threshold(
    positives: Sequence[float], negatives: Sequence[float]
) -> float | None:
    """The cut maximising sensitivity + specificity - 1.

    None when either side is empty: there is nothing to separate, and returning a number
    would imply a calibration that never happened.

    Candidates are the observed scores themselves, so the chosen cut is always one the
    data actually supports rather than an interpolation between gaps.
    """
    if not positives or not negatives:
        return None

    best_cut = 0.0
    best_j = float("-inf")
    for candidate in sorted({*positives, *negatives}):
        sensitivity = sum(1 for p in positives if p >= candidate) / len(positives)
        specificity = sum(1 for n in negatives if n < candidate) / len(negatives)
        j = sensitivity + specificity - 1
        if j > best_j:
            best_j, best_cut = j, candidate
    return best_cut


def rates(
    positives: Sequence[float], negatives: Sequence[float], threshold: float
) -> dict[str, float]:
    """Sensitivity, specificity and balanced accuracy at a given cut."""
    sensitivity = (
        sum(1 for p in positives if p >= threshold) / len(positives)
        if positives
        else 0.0
    )
    specificity = (
        sum(1 for n in negatives if n < threshold) / len(negatives)
        if negatives
        else 0.0
    )
    return {
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "balanced_accuracy": round((sensitivity + specificity) / 2, 4),
    }


def chunk_covering(span: dict, chunks: Sequence[Any]) -> Any | None:
    """The chunk whose character range contains a labelled evidence span.

    `relevant_chunk_ids` lists every chunk touching a question, and only some of them
    carry the sentence the answer rests on. Taking the first pairs a claim with a passage
    that does not support it, which looks like a judge failure and is a labelling choice.
    The ground truth records exact offsets; this uses them.
    """
    document = span.get("document_id")
    start, end = span.get("start"), span.get("end")
    if document is None or start is None or end is None:
        return None
    for chunk in chunks:
        if (
            getattr(chunk, "document_id", None) == document
            and chunk.start <= start
            and end <= chunk.end
        ):
            return chunk
    return None


def paraphrase_control(
    rows: Sequence[dict],
    texts: dict[str, str],
    chunks: Sequence[Any] | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """A control shaped like the thing it calibrates.

    The first control here paired each block with a sentence lifted verbatim out of it.
    That separates cleanly - the judge scored 30 of 30 - and it calibrates the wrong
    task. Verbatim text scores near 1.0, so Youden's J put the cut at 0.98, and a
    citation metric does not score copies: an answer paraphrases its source, and a
    paraphrase legitimately scores lower. A cut fitted to copies is far too strict for
    the pairs actually being measured, which is why recall fell by half when it was
    applied.

    This builds positives from the ground truth instead: a human-written `gold_answer`
    against a chunk its labels mark relevant. That is a paraphrase supported by a
    passage, which is exactly the shape of a citation. Negatives pair the same answer
    with a chunk the labels do not mark relevant.

    Unanswerable questions are skipped; they carry no gold answer, so no chunk can
    support one.
    """
    positives: list[tuple[str, str]] = []
    negatives: list[tuple[str, str]] = []

    for row in rows:
        answer = (row.get("gold_answer") or "").strip()
        if not row.get("answerable") or not answer:
            continue
        # Prefer the chunk carrying the labelled evidence; fall back to the first
        # relevant one only when no span is recorded.
        premise = None
        if chunks:
            for span in row.get("evidence_spans") or []:
                found = chunk_covering(span, chunks)
                if found is not None:
                    premise = found.text
                    break
        if premise is None:
            relevant = [c for c in row.get("relevant_chunk_ids", []) if c in texts]
            premise = texts[relevant[0]] if relevant else None
        if premise is not None:
            positives.append((premise, answer))
        irrelevant = [
            c for c in texts if c not in set(row.get("relevant_chunk_ids", []))
        ]
        for chunk_id in irrelevant[:1]:
            negatives.append((texts[chunk_id], answer))

    return positives, negatives


__all__ = [
    "Scorer",
    "batch_plan",
    "best_scores",
    "best_sentence_score",
    "paraphrase_control",
    "premise_parts",
    "rates",
    "youden_threshold",
]
