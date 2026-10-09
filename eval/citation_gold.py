"""A human-labelled gold standard for citation correctness, and the judge scored on it.

`eval/citation_correctness_run.py` reports ALCE recall and precision from an NLI judge
calibrated on *(chunk, gold_answer)* pairs. The pairs it actually scores are *(cited
blocks concatenated, generated sentence)*. Those are different populations - the premise is
assembled differently and the hypothesis has a different author - so the judge's measured
sensitivity does not transfer to them, and nothing in that run establishes the judge agrees
with a person on the pairs that produced the published number.

ALCE does establish it, for its own judge: human annotation of 100 examples per dataset,
reported as Cohen's kappa of 0.698 for citation recall and 0.525 for precision (Gao et al.,
arXiv:2305.14627). This module is the same step at this corpus's scale. There are 16 cited
sentences, so labelling every pair by hand costs less than any statistical correction and
answers both questions at once: what the rate actually is, and whether the judge can be
trusted to scale it to more questions later.

Why not debias the judge instead. Prediction-powered inference (Angelopoulos et al.,
Science 2023) corrects a machine-labelled rate using a labelled subset, and its validity
rests on that subset being drawn from the target population. The calibration set here is
not, which is the same mismatch described above. Hand labels fix the mismatch; they are
also the precondition for using PPI later, once there are more pairs than one person
should read.

    python -m eval.citation_gold_run --build    # writes the worksheet to fill in
    python -m eval.citation_gold_run --score    # reads the labels, writes the artefact

The judgments a person makes are deliberately the two ALCE defines, and no others:
whether the cited blocks support the sentence, and - for a sentence citing more than one
block - whether the remaining blocks still support it without each one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eval.citation_correctness import claims, hypothesis_of, is_claim

ROOT = Path(__file__).resolve().parents[1]

#: The two kinds of judgment. `SUPPORT` is recall's question, `NECESSITY` is precision's.
SUPPORT = "support"
NECESSITY = "necessity"

GOLD_DIR = ROOT / "eval" / "citation_gold"
LABELS = GOLD_DIR / "labels.jsonl"
PAIRS = GOLD_DIR / "pairs.jsonl"
WORKSHEET = GOLD_DIR / "worksheet.md"


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> float | None:
    """Chance-corrected agreement between two binary raters.

    Raw agreement flatters a skewed set. Most cited sentences here are unsupported, so a
    judge that answers "unsupported" every time is right most of the time and has no
    skill; kappa is what says so, and it is the statistic ALCE reports.

    None when expected agreement is 1 - both raters constant and identical. The
    correction divides by zero there, and a set with no disagreement to resolve carries
    no evidence of skill either way.
    """
    if len(a) != len(b):
        raise ValueError(f"rater lengths differ: {len(a)} and {len(b)}")
    n = len(a)
    if n == 0:
        return None

    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa, pb = sum(1 for x in a if x) / n, sum(1 for y in b if y) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


def agreement(human: Sequence[bool], judge: Sequence[bool]) -> dict[str, Any]:
    """The judge measured against the human labels, treating the human as truth.

    This is the sensitivity and specificity the calibration in
    `eval/citation_correctness_run.py` could not provide, because it was measured on a
    different population. Counts are reported alongside the rates so both can be checked.
    """
    if len(human) != len(judge):
        raise ValueError(f"label counts differ: {len(human)} and {len(judge)}")

    tp = sum(1 for h, j in zip(human, judge, strict=True) if h and j)
    fn = sum(1 for h, j in zip(human, judge, strict=True) if h and not j)
    tn = sum(1 for h, j in zip(human, judge, strict=True) if not h and not j)
    fp = sum(1 for h, j in zip(human, judge, strict=True) if not h and j)
    kappa = cohen_kappa(human, judge)

    return {
        "n": len(human),
        "human_supported": tp + fn,
        "judge_supported": tp + fp,
        "both_supported": tp,
        "sensitivity": tp / (tp + fn) if (tp + fn) else None,
        "specificity": tn / (tn + fp) if (tn + fp) else None,
        "kappa": kappa,
        "kappa_note": (
            "Compare ALCE's human agreement for its own judge: 0.698 for citation "
            "recall, 0.525 for precision (Gao et al., arXiv:2305.14627)."
        ),
    }


def _premise(blocks: Mapping[int, str], citations: Sequence[int]) -> str:
    """ALCE's concat(C_i), identical to `eval.citation_correctness._premise`.

    Duplicated rather than imported because it is private there, and because a worksheet
    showing text the metric never scored would collect labels for a different
    measurement. The test suite pins the two against each other.
    """
    return "\n".join(blocks[n] for n in citations if n in blocks)


def build_blocks() -> dict[str, dict[int, str]]:
    """The numbered context blocks each question was shown, rebuilt from the reference
    run exactly as the correctness run rebuilds them."""
    from src.corpus import chunk_corpus, load_corpus

    reference = json.loads(
        (ROOT / "eval" / "reference_run.json").read_text(encoding="utf-8")
    )
    chunking = reference["retrieval"]["chunking"]
    top_k = int(reference["retrieval"]["serving_top_k"])
    by_id = {
        c.chunk_id: c
        for c in chunk_corpus(load_corpus(), chunking["size"], chunking["overlap"])
    }
    return {
        q["question_id"]: {
            i: by_id[cid].text
            for i, cid in enumerate(q["retrieved_chunk_ids"][:top_k], start=1)
        }
        for q in reference["per_question"]
    }


def gold_pairs(
    structure: Mapping[str, Any], blocks_by_question: Mapping[str, Mapping[int, str]]
) -> list[dict[str, Any]]:
    """Every judgment a person has to make, in a stable order with stable ids.

    Sentences citing a single block get no necessity pair: dropping the sole citation
    leaves nothing to support the claim, so it is necessary by construction and asking is
    wasted effort. This matches how `citation_precision` already treats them.
    """
    out: list[dict[str, Any]] = []
    for row in structure["per_question"]:
        qid = row["question_id"]
        blocks = blocks_by_question.get(qid, {})
        for index, claim in enumerate(
            claims(row["answer"], n_contexts=row["n_contexts"]), start=1
        ):
            hypothesis = hypothesis_of(claim.sentence)
            # A hypothesis that asserts nothing cannot be labelled: there is no statement
            # to agree or disagree with. Those pairs are counted by
            # `excluded_non_claims` instead of being put to a person.
            if not is_claim(hypothesis):
                continue
            out.append(
                {
                    "pair_id": f"{qid}#s{index}",
                    "kind": SUPPORT,
                    "question_id": qid,
                    "sentence_index": index,
                    "citations": list(claim.citations),
                    "premise": _premise(blocks, claim.citations),
                    "hypothesis": hypothesis,
                    "question": (
                        "Do these blocks, taken together, support the statement?"
                    ),
                }
            )
            if len(claim.citations) < 2:
                continue
            for dropped in claim.citations:
                rest = [n for n in claim.citations if n != dropped]
                out.append(
                    {
                        "pair_id": f"{qid}#s{index}-drop{dropped}",
                        "kind": NECESSITY,
                        "question_id": qid,
                        "sentence_index": index,
                        "citations": list(claim.citations),
                        "dropped": dropped,
                        "remaining": rest,
                        "premise": _premise(blocks, rest),
                        "hypothesis": hypothesis,
                        "question": (
                            f"With block [{dropped}] removed, do the remaining blocks "
                            "still support the statement?"
                        ),
                    }
                )
    return out


def excluded_non_claims(structure: Mapping[str, Any]) -> dict[str, Any]:
    """How many cited sentences the worksheet leaves out, and why.

    The generator sometimes places its citations in subject position - "[1], [2], and
    provide relevant information." - so removing the markers leaves a fragment, or in one
    case nothing at all. Those cannot be labelled and cannot be entailed, so a
    correctness metric that includes them is partly scoring citation placement. The count
    is surfaced so the exclusion is visible rather than inferred from a denominator.
    """
    total = excluded = 0
    examples: list[str] = []
    for row in structure["per_question"]:
        for claim in claims(row["answer"], n_contexts=row["n_contexts"]):
            total += 1
            hypothesis = hypothesis_of(claim.sentence)
            if not is_claim(hypothesis):
                excluded += 1
                examples.append(hypothesis or "(empty)")
    return {
        "excluded": excluded,
        "of": total,
        "why": (
            "Nothing survived removing the citation markers, so there is no statement to "
            "label. Fragments left by citations in subject position are NOT excluded - a "
            "person can read those and decide."
        ),
        "examples": examples[:5],
    }


def load_labels(path: Path | str = LABELS) -> dict[str, bool]:
    """Read `pair_id -> supported` from a JSONL file of human labels.

    A row without a `supported` field is an error rather than a False, and an explicit
    `null` is skipped as unanswered. The worksheet ships pre-seeded with nulls so the
    annotator edits values instead of writing JSONL, and `bool(None)` is False - reading
    null as a label would turn every untouched row into evidence that the citation is
    unsupported, and let a half-finished pass publish a rate.
    """
    path = Path(path)
    if not path.exists():
        return {}

    labels: dict[str, bool] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        row = json.loads(text)
        if "supported" not in row:
            raise ValueError(f"{path}:{number}: row has no 'supported' field: {text}")
        if row["supported"] is None:
            continue
        labels[row["pair_id"]] = bool(row["supported"])
    return labels


def human_counts(
    pairs: Sequence[Mapping[str, Any]], labels: Mapping[str, bool]
) -> dict[str, Any]:
    """ALCE recall and precision from the human labels.

    Both are None until every support judgment is in. A rate over half the sentences is
    a different quantity wearing the same name, and publishing it as progress is how a
    partial pass becomes a headline.
    """
    support = [p for p in pairs if p["kind"] == SUPPORT]
    unlabelled = sum(1 for p in pairs if p["pair_id"] not in labels)
    missing_support = [p for p in support if p["pair_id"] not in labels]

    result: dict[str, Any] = {
        "pairs": len(pairs),
        "support_pairs": len(support),
        "unlabelled": unlabelled,
        "recall": None,
        "precision": None,
    }
    if missing_support:
        return result

    supported = {p["pair_id"]: labels[p["pair_id"]] for p in support}
    result["recall"] = {
        "supported": sum(1 for v in supported.values() if v),
        "cited": len(support),
    }

    # Precision apportions attribution only on sentences that are supported at all.
    needed = offered = 0
    for p in support:
        if not supported[p["pair_id"]]:
            continue
        if len(p["citations"]) < 2:
            offered += 1
            needed += 1  # a sole citation is necessary by construction
            continue
        for dropped in p["citations"]:
            key = f"{p['question_id']}#s{p['sentence_index']}-drop{dropped}"
            if key not in labels:
                # A supported multi-citation sentence with an unfinished drop-one pass
                # cannot contribute a precision figure.
                return {**result, "precision": None}
            offered += 1
            if not labels[key]:
                needed += 1
    result["precision"] = {"needed": needed, "offered": offered}
    return result


__all__ = [
    "LABELS",
    "NECESSITY",
    "PAIRS",
    "SUPPORT",
    "WORKSHEET",
    "agreement",
    "build_blocks",
    "cohen_kappa",
    "excluded_non_claims",
    "gold_pairs",
    "human_counts",
    "load_labels",
]
