"""Measure citation placement on both pinned prompt variants.

    python -m eval.citation_placement_run

Offline, deterministic, no model and no judge: both answer sets are already pinned in
`eval/citation_structure.json` and `eval/citation_structure_ranged.json`, and every
quantity here is a fact about those strings. Writes `eval/citation_placement.json`.

The paired comparison only uses questions that produced at least one cited sentence under
*both* prompts. Three of the four raw differences between the variants are questions where
the served answer cited nothing at all, and an answer with no citations has not placed its
citations well - it has no citations. Counting those as a win for the served prompt would
reward silence.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.citation_placement import mcnemar_exact, placement_counts

OUT = ROOT / "eval" / "citation_placement.json"
VARIANTS = {
    "served": ROOT / "eval" / "citation_structure.json",
    "ranged": ROOT / "eval" / "citation_structure_ranged.json",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, as in scripts/figures/generate_figures.py.

    Normal approximations are wrong at these counts, and one of the proportions here is
    16/16, where a normal interval would have zero width and claim certainty.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate(k: int, n: int) -> dict[str, Any]:
    low, high = wilson(k, n)
    return {
        "count": k,
        "of": n,
        "rate": round(k / n, 4) if n else None,
        "wilson_95": [round(low, 3), round(high, 3)],
    }


def main() -> int:
    counts = {
        name: placement_counts(json.loads(path.read_text(encoding="utf-8")))
        for name, path in VARIANTS.items()
    }

    variants = {
        name: {
            "cited_sentences": c["cited_sentences"],
            "leading": _rate(c["leading"], c["cited_sentences"]),
            "coordinated": _rate(c["coordinated"], c["cited_sentences"]),
        }
        for name, c in counts.items()
    }

    # Paired on questions that cited something under both prompts.
    served, ranged = counts["served"]["per_question"], counts["ranged"]["per_question"]
    comparable = sorted(
        q
        for q in set(served) & set(ranged)
        if served[q]["any_leading"] is not None and ranged[q]["any_leading"] is not None
    )
    excluded = sorted(
        q
        for q in set(served) & set(ranged)
        if served[q]["any_leading"] is None or ranged[q]["any_leading"] is None
    )

    paired: dict[str, Any] = {}
    for label, key in (("leading", "any_leading"), ("coordinated", "any_coordinated")):
        only_served = [q for q in comparable if served[q][key] and not ranged[q][key]]
        only_ranged = [q for q in comparable if ranged[q][key] and not served[q][key]]
        paired[label] = {
            "questions": len(comparable),
            "both": sum(1 for q in comparable if served[q][key] and ranged[q][key]),
            "neither": sum(
                1 for q in comparable if not served[q][key] and not ranged[q][key]
            ),
            "served_only": only_served,
            "ranged_only": only_ranged,
            "p_mcnemar_exact": round(
                mcnemar_exact(len(only_served), len(only_ranged)), 4
            ),
        }

    payload = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "measures": (
            "Where the generator puts its citation markers. 'leading' = the sentence "
            "opens with a marker before any prose; 'coordinated' = markers joined by a "
            "conjunction, so they serve as a noun phrase."
        ),
        "why": (
            "Citation correctness was first reported from malformed hypotheses, because "
            "removing leading markers stranded punctuation at the front of the sentence. "
            "Repairing that moved recall from 0.3125 to 0.4667. This asks whether the "
            "placement is incidental or systematic."
        ),
        "judge": "none - every quantity here is a fact about the pinned answer strings",
        "variants": variants,
        "paired": paired,
        "paired_note": (
            "Questions that cited nothing under either prompt are excluded from the "
            "pairing: an answer with no citations has not placed them well, and counting "
            "it as a win would reward silence."
        ),
        "paired_excluded": excluded,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for name, v in variants.items():
        lead, coord = v["leading"], v["coordinated"]
        print(
            f"{name:8s} leading {lead['count']}/{lead['of']} = {lead['rate']} "
            f"{lead['wilson_95']}   coordinated {coord['count']}/{coord['of']} "
            f"= {coord['rate']} {coord['wilson_95']}"
        )
    for label, p in paired.items():
        print(
            f"paired {label:12s} n={p['questions']} both={p['both']} "
            f"neither={p['neither']} served_only={len(p['served_only'])} "
            f"ranged_only={len(p['ranged_only'])} p={p['p_mcnemar_exact']}"
        )
    print(f"excluded from pairing: {', '.join(excluded) or 'none'}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
