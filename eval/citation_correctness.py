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


def hypothesis_of(sentence: str) -> str:
    """The sentence as a claim: citation markers removed, punctuation tidied."""
    stripped = _MARKER_SPAN.sub("", sentence or "").strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", stripped)


def _premise(blocks: Mapping[int, str], citations: Sequence[int]) -> str:
    return "\n".join(blocks[n] for n in citations if n in blocks)


def citation_recall(
    items: Sequence[Claim], blocks: Mapping[int, str], entails: Entails
) -> float | None:
    """Share of cited sentences entailed by the blocks they cite, together.

    None when nothing was cited: there is no recall to report, and zero would read as
    total failure rather than as absence of evidence.
    """
    if not items:
        return None
    supported = sum(
        1
        for c in items
        if entails(_premise(blocks, c.citations), hypothesis_of(c.sentence))
    )
    return supported / len(items)


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
    return needed / total if total else None


__all__ = [
    "Claim",
    "citation_precision",
    "citation_recall",
    "claims",
    "hypothesis_of",
    "split_sentences",
]
