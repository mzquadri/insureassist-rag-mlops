"""Whether a cited block actually supports the sentence it is attached to.

`eval/citation_structure.py` asks whether a citation is structurally real — did the model
cite, and did it cite a block it was given. This asks the next question: does the block
say what the sentence claims.

The definitions are ALCE's (Gao et al., arXiv:2305.14627). **Citation recall** is the
share of cited sentences whose citations, taken together, entail the sentence.
**Citation precision** is the share of individual citations that were needed for that —
drop one, and if the rest still entail, that citation was along for the ride.

This is the question `docs/LIMITATIONS.md` says is unmeasured, and the circularity
objection is why: the only local model grading its own output proves nothing. ALCE's
answer is to judge with an entailment model instead, which is a different model doing a
different task. Entailment arrives here as a callable so the arithmetic is testable
without one, and so the judge can be swapped without touching the metric.

**What this is not.** ALCE validated against the TRUE NLI model and reported Cohen's
kappa of 0.698 for recall; a different, smaller judge does not inherit that agreement.
And correctness is not faithfulness: that a cited block supports a claim does not
establish the model derived the claim from it rather than from memory and then found a
passage that happened to fit.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from eval.citation_structure import parse_citations

#: Split after a terminator only when what follows looks like a new sentence: whitespace
#: then a capital, a digit, or an opening citation. Policy text is full of amounts, and
#: splitting inside "$30,000.00" would cut a claim in half.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(])")

Entails = Callable[[str, str], bool]


@dataclass(frozen=True)
class Claim:
    """One sentence and the blocks it cites."""

    sentence: str
    citations: list[int]


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_END.split(text or "") if part.strip()]


def claims(answer: str, n_contexts: int) -> list[Claim]:
    """Cited sentences only, with out-of-range citations dropped.

    An uncited sentence has no citation to be right or wrong about; counting it would
    fold abstention into a citation metric. Out-of-range numbers are already reported as
    structural failures, and scoring them again here would punish one mistake twice.
    """
    out: list[Claim] = []
    for sentence in split_sentences(answer):
        cited = [n for n in parse_citations(sentence) if 1 <= n <= n_contexts]
        if cited:
            out.append(Claim(sentence=sentence, citations=cited))
    return out


#: Citation markers are notation, not part of the claim. Leaving "[2]" in the hypothesis
#: hands the judge a token that appears in no policy text and belongs to no assertion.
_MARKER_SPAN = re.compile(r"\s*\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]")


#: Debris left at the front when the markers were part of the sentence rather than
#: attached to it. The generator cites in subject position - "[1], [2] and provide
#: relevant information." - so removing the markers strands the punctuation and the
#: conjunction that joined them. Both are notation's remains, not part of the claim.
_LEADING_DEBRIS = re.compile(r"^(?:[\s,;:.\-]|\b(?:and|or|but|nor|as well as)\b)+")


def hypothesis_of(sentence: str) -> str:
    """The sentence as a claim: citation markers removed, whitespace and debris tidied.

    Internal newlines are collapsed because the generator emits its citations as a
    preamble - "[2], [5]\\n\\nThe maximum payable is..." - and a hypothesis carrying a
    paragraph break is not the sentence anyone meant to score. Leading punctuation and a
    leading conjunction go for the same reason: they are what the markers left behind.

    Stripping the conjunction rather than rejecting the sentence matters. An earlier
    version treated any remainder starting with "and" as unscoreable, which discarded
    "and According to the context, ... the revised due date will be 30 days after the
    date on which the bill is mailed" - a real claim with a connective stuck to the
    front. That was a hand-made grammar rule getting the grammar wrong.
    """
    stripped = _MARKER_SPAN.sub("", sentence or "")
    collapsed = re.sub(r"\s+", " ", stripped).strip()
    tidied = re.sub(r"\s+([.,;:!?])", r"\1", collapsed)
    return _LEADING_DEBRIS.sub("", tidied).strip()


def is_claim(hypothesis: str) -> bool:
    """Whether there is anything left to judge.

    Deliberately the weakest possible test: non-empty. A sentence whose entire content
    was its citations - "[1] [2] [3]" - leaves nothing, and neither a person nor an
    entailment model can rule on nothing; including it in a correctness rate scores the
    generator's citation placement, which `eval/citation_structure.py` already reports.

    Everything else goes to the annotator, including fragments like "provide relevant
    information" that assert something about the blocks rather than about flood policy.
    Those are judgments about meaning, and the point of the human pass is that a person
    makes them. A regex deciding instead is how the previous version lost a real claim.
    """
    return bool((hypothesis or "").strip())


def _premise(blocks: Mapping[int, str], citations: Sequence[int]) -> str:
    return "\n".join(blocks[n] for n in citations if n in blocks)


def citation_recall_counts(
    items: Sequence[Claim], blocks: Mapping[int, str], entails: Entails
) -> tuple[int, int]:
    """(supported, cited) behind the recall rate.

    The rate alone cannot carry a confidence interval, and 5/16 and 50/160 are the same
    number on very different evidence. This arm's denominators are small enough that the
    distinction decides whether the figure supports a conclusion, so the counts are
    returned and recorded rather than reconstructed from a rounded rate.
    """
    supported = sum(
        1
        for c in items
        if entails(_premise(blocks, c.citations), hypothesis_of(c.sentence))
    )
    return supported, len(items)


def citation_recall(
    items: Sequence[Claim], blocks: Mapping[int, str], entails: Entails
) -> float | None:
    """Share of cited sentences entailed by the blocks they cite, together.

    None when nothing was cited: there is no recall to report, and zero would read as
    total failure rather than as absence of evidence.
    """
    supported, total = citation_recall_counts(items, blocks, entails)
    return supported / total if total else None


def citation_precision(
    items: Sequence[Claim], blocks: Mapping[int, str], entails: Entails
) -> float | None:
    """Share of individual citations that were needed.

    Only sentences that are supported at all contribute. On an unsupported sentence there
    is no attribution to apportion, and counting its citations as imprecise would conflate
    two different failures.

    A sole citation on a supported sentence is necessary by construction: removing it
    leaves nothing to entail from.
    """
    needed, total = citation_precision_counts(items, blocks, entails)
    return needed / total if total else None


def citation_precision_counts(
    items: Sequence[Claim], blocks: Mapping[int, str], entails: Entails
) -> tuple[int, int]:
    """(needed, offered) behind the precision rate.

    Recorded for the same reason as the recall counts: precision's denominator is the
    number of citations on supported sentences, which is smaller than the sentence count
    and is not recoverable from the rate.
    """
    total = 0
    needed = 0
    for claim in items:
        hypothesis = hypothesis_of(claim.sentence)
        if not entails(_premise(blocks, claim.citations), hypothesis):
            continue
        for citation in claim.citations:
            total += 1
            rest = [n for n in claim.citations if n != citation]
            if not rest or not entails(_premise(blocks, rest), hypothesis):
                needed += 1
    return needed, total


__all__ = [
    "Claim",
    "citation_precision",
    "citation_precision_counts",
    "citation_recall",
    "citation_recall_counts",
    "claims",
    "hypothesis_of",
    "is_claim",
    "split_sentences",
]
