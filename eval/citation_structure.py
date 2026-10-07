"""Did the generator use the citations it was told to use?

The prompt numbers each context block and asks the model to "cite the numbers you used
in square brackets". `docs/LIMITATIONS.md` records that whether it does so is not
measured, and that no answer-quality metric is published because the only local model
grading its own output would be circular.

The circularity objection is right, and it does not reach this far. Nothing here judges
an answer. These are facts about the string: did it cite at all, did it cite a block
that was actually supplied, how much of the supplied context did it lean on. A parser
has no opinion to be circular with.

What this does *not* measure is whether a cited block supports the sentence it is
attached to. That is citation correctness, it needs an entailment model, and it is a
separate question from whether the citation is structurally real.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

#: A bracketed group of one or more integers: [1], [1, 2], [1][2].
#: Prose in brackets - "[see section 3]" - is not a citation and must not be counted,
#: so the whole bracket has to be digits and separators or it is ignored.
_MARKER = re.compile(r"\[\s*(\d+(?:\s*[,;]\s*\d+)*)\s*\]")


def parse_citations(answer: str) -> list[int]:
    """Every context number the answer refers to, in order of first appearance."""
    seen: list[int] = []
    for group in _MARKER.findall(answer or ""):
        for part in re.split(r"[,;]", group):
            number = int(part.strip())
            if number not in seen:
                seen.append(number)
    return seen


@dataclass(frozen=True)
class CitationUse:
    """What one answer did with the context it was given."""

    cited: list[int] = field(default_factory=list)
    in_range: list[int] = field(default_factory=list)
    out_of_range: list[int] = field(default_factory=list)
    n_contexts: int = 0

    @property
    def cited_any(self) -> bool:
        return bool(self.cited)

    @property
    def context_coverage(self) -> float:
        """Share of the supplied blocks the answer actually pointed at.

        Zero contexts returns zero rather than dividing: there is no coverage to report,
        and a fabricated rate would average into the summary as though it were one.
        """
        if self.n_contexts <= 0:
            return 0.0
        return len(self.in_range) / self.n_contexts

    def as_dict(self) -> dict[str, Any]:
        return {
            "cited": self.cited,
            "in_range": self.in_range,
            "out_of_range": self.out_of_range,
            "n_contexts": self.n_contexts,
            "cited_any": self.cited_any,
            "context_coverage": round(self.context_coverage, 4),
        }


def use_of_citations(answer: str, n_contexts: int) -> CitationUse:
    """Split an answer's citations into the ones that exist and the ones that do not.

    Blocks are numbered from 1, so 0 and anything above `n_contexts` point at context
    the model was never given.
    """
    cited = parse_citations(answer)
    in_range = [n for n in cited if 1 <= n <= n_contexts]
    out_of_range = [n for n in cited if n < 1 or n > n_contexts]
    return CitationUse(
        cited=cited,
        in_range=in_range,
        out_of_range=out_of_range,
        n_contexts=n_contexts,
    )


def summarise(rows: Sequence[CitationUse]) -> dict[str, Any]:
    """Counts over answers, not over markers.

    One answer citing a fabricated block four times is one answer with a problem, not
    four. Rates that counted markers would move with verbosity.
    """
    total = len(rows)
    if total == 0:
        return {
            "answers": 0,
            "cited_any": 0,
            "uncited": 0,
            "with_out_of_range": 0,
            "uncited_rate": 0.0,
            "out_of_range_rate": 0.0,
            "mean_context_coverage": 0.0,
            # Percentages as well as fractions, because the docs quote percentages and
            # eval/verify_artifacts.py matches the documented text against this file.
            # Publishing only fractions would check "27.8%" against 0.278 and so never
            # check it at all.
            "uncited_pct": 0.0,
            "out_of_range_pct": 0.0,
            "mean_context_coverage_pct": 0.0,
        }

    cited_any = sum(1 for r in rows if r.cited_any)
    out_of_range = sum(1 for r in rows if r.out_of_range)
    coverage = sum(r.context_coverage for r in rows) / total
    return {
        "answers": total,
        "cited_any": cited_any,
        "uncited": total - cited_any,
        "with_out_of_range": out_of_range,
        "uncited_rate": round((total - cited_any) / total, 4),
        "out_of_range_rate": round(out_of_range / total, 4),
        "mean_context_coverage": round(coverage, 4),
        "uncited_pct": round(100 * (total - cited_any) / total, 1),
        "out_of_range_pct": round(100 * out_of_range / total, 1),
        "mean_context_coverage_pct": round(100 * coverage, 1),
    }


__all__ = ["CitationUse", "parse_citations", "summarise", "use_of_citations"]
