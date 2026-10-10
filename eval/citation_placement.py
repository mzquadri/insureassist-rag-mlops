"""Where the generator puts its citation markers.

This arm exists because of a defect it explains. Citation correctness was first reported
from malformed hypotheses: the model emits `"[2], [5]\\n\\nThe maximum payable ... is
$30,000."`, so removing the markers left punctuation stranded at the front of nine of
sixteen sentences, and one sentence had no content at all. Repairing the extraction moved
recall from 0.3125 to 0.4667 - a quarter of the published figure had been an artefact of
reading the string badly.

That fix could not answer the question it raised: was the placement incidental, or is it
how this model cites? This measures it. **No judge is involved** - whether a sentence
begins with a marker is decidable by reading characters, which is why this survives the
circularity objection in `docs/LIMITATIONS.md` that blocks answer-quality scoring here, the
same way the dangling-citation rate in `eval/citation_structure.py` does.

Why placement is not cosmetic. The `[n]` convention, and ALCE's metric built on it, assume
the marker *follows* the statement it supports; that is what makes "which claim does this
citation back" answerable. A sentence opening with its citations leaves the attachment
ambiguous to a reader and to any metric, and a system whose citations resolve to exact
character offsets - as this one advertises - is making a weaker promise than it appears to
if the offsets are not attached to a particular claim.

Two patterns are counted, separately, because a sentence can show either, both or neither:

**leading** - the sentence opens with a marker, before any prose.
**coordinated** - markers joined by a conjunction, "[2], [3], and [4] provide relevant
information", so the citations are serving as a noun phrase rather than an attachment.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from eval.citation_correctness import claims

#: A marker at the very start, after optional whitespace. Requires digits inside the
#: brackets: "[see below]" is prose, not a citation, and `parse_citations` would not read
#: it as one either.
_LEADS = re.compile(r"^\s*\[\s*\d+\s*\]")

#: Markers joined by a conjunction, with an optional comma before it - the Oxford comma in
#: "[2], [3], and [4]". Both markers are required, so "Coverage A and Coverage D differ.
#: [1]" does not match: the conjunction has to join citations, not words.
_COORDINATED = re.compile(r"\]\s*(?:,\s*)?(?:and|or)\s*\[", re.IGNORECASE)


def leads_with_citation(sentence: str) -> bool:
    """Whether the sentence opens with a citation marker rather than ending with one."""
    return bool(_LEADS.match(sentence or ""))


def coordinates_citations(sentence: str) -> bool:
    """Whether two markers are joined by a conjunction, making them a noun phrase."""
    return bool(_COORDINATED.search(sentence or ""))


def placement_counts(structure: Mapping[str, Any]) -> dict[str, Any]:
    """Leading and coordinated counts over one artefact, per question and in total.

    `any_leading` is None for a question that cited nothing. Absence of citations is not
    good placement, and pairing a question that cited nothing against one that cited well
    is the confound that makes a raw served-versus-ranged comparison misleading: three of
    the four apparent differences between the two prompts are questions where the served
    answer produced no cited sentence at all.
    """
    per_question: dict[str, dict[str, Any]] = {}
    leading = coordinated = cited = 0

    for row in structure["per_question"]:
        items = claims(row["answer"], n_contexts=row["n_contexts"])
        lead = sum(1 for c in items if leads_with_citation(c.sentence))
        coord = sum(1 for c in items if coordinates_citations(c.sentence))
        leading += lead
        coordinated += coord
        cited += len(items)
        per_question[row["question_id"]] = {
            "cited_sentences": len(items),
            "leading": lead,
            "coordinated": coord,
            "any_leading": bool(lead) if items else None,
            "any_coordinated": bool(coord) if items else None,
        }

    return {
        "cited_sentences": cited,
        "leading": leading,
        "coordinated": coordinated,
        "per_question": per_question,
    }


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test on `b` and `c` discordant pairs.

    The concordant pairs carry no information about a difference, so the test asks only
    whether the discordant ones split evenly: under the null, each is a fair coin. The
    exact binomial is used rather than the chi-square approximation because the counts
    here are single digits, which is also why `docs/LIMITATIONS.md` quotes an exact p for
    the prompt comparison. That figure, p = 0.250 on three one-way pairs, was computed by
    hand; this makes it reproducible.
    """
    n = b + c
    if n == 0:
        return 1.0

    from math import comb

    smaller = min(b, c)
    tail = sum(comb(n, k) for k in range(smaller + 1)) / (2**n)
    return min(1.0, 2 * tail)


__all__ = [
    "coordinates_citations",
    "leads_with_citation",
    "mcnemar_exact",
    "placement_counts",
]
