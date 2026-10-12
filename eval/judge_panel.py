"""The spread between entailment judges, recorded as data.

Two judges clear the same calibration floors - both declared before either was run - and
differ on ALCE citation recall by a third, 0.4667 against 0.8000. **That spread is the
dominant uncertainty in this arm**, wider than the sampling interval around either figure,
and it was living only in a paragraph. Prose drifts away from artefacts; this does not,
because `eval/verify_artifacts.py` pins it.

Reporting a panel rather than picking a winner is the practice the attribution-evaluation
literature arrived at. AttributionBench (Li et al., arXiv:2402.15089) finds no single
automatic evaluator suffices, and that evaluators are weakest in precisely this regime:
long evidence, an expert domain, and claims that turn on numbers and dates - 66% of their
error analysis is fine-grained insensitivity to exactly those. An 800-character block of
flood-policy prose asking whether the limit is $30,000 or the deadline 60 days is the hard
case, not an unlucky one.

What this module deliberately does **not** do is choose. Only the sensitivity and
specificity floors were declared in advance; balanced accuracy as a *selection* rule was
not, and applying it after seeing which judge gives the friendlier recall is the post-hoc
choice the floors exist to prevent. `eval/citation_gold.py` adjudicates, with human labels.

    python -m eval.judge_panel_run

Offline and deterministic: both artefacts already exist, and every number here is derived
from them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from eval.citation_placement import mcnemar_exact


def interval_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Width of the region two confidence intervals share.

    Zero would say the two judges cannot both be right. A wide overlap would say the
    disagreement is sampling noise and not worth a paragraph. Here it is neither, which
    is the reason to report the number rather than describe it.
    """
    low = max(a[0], b[0])
    high = min(a[1], b[1])
    return max(0.0, high - low)


def _verdicts(artefact: Mapping[str, Any]) -> dict[str, bool]:
    return {
        f"{row['question_id']}#s{s['index']}": bool(s["supported"])
        for row in artefact.get("per_question", [])
        for s in (row.get("sentences") or [])
    }


def panel_summary(artefacts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Each judge's calibration and rates, and the paired comparison between them.

    `paired` is None with fewer than two judges: there is no panel, and reporting a
    spread of zero would read as agreement rather than as absence of a comparison.
    """
    judges = {
        name: {
            "model": a["judge"].get("model"),
            "calibration": a["calibration"],
            "claims_only": a["claims_only"],
        }
        for name, a in artefacts.items()
    }

    summary: dict[str, Any] = {"judges": judges, "paired": None, "recall_spread": None}
    if len(artefacts) < 2:
        return summary

    (first, a), (second, b) = list(artefacts.items())[:2]
    summary["recall_spread"] = abs(
        a["claims_only"]["citation_recall"] - b["claims_only"]["citation_recall"]
    )

    va, vb = _verdicts(a), _verdicts(b)
    shared = sorted(va.keys() & vb.keys())
    only_a = [k for k in shared if va[k] and not vb[k]]
    only_b = [k for k in shared if vb[k] and not va[k]]

    summary["paired"] = {
        "sentences": len(shared),
        "both": sum(1 for k in shared if va[k] and vb[k]),
        "neither": sum(1 for k in shared if not va[k] and not vb[k]),
        f"only_{first}": len(only_a),
        f"only_{second}": len(only_b),
        "contested": sorted(only_a + only_b),
        # One judge strictly stricter than the other is a different situation from two
        # judges each making their own mistakes: it points at a threshold, not at noise,
        # and it means their errors are not independent so averaging them buys nothing.
        "nested": not (only_a and only_b),
        "p_mcnemar_exact": round(mcnemar_exact(len(only_a), len(only_b)), 4),
    }
    return summary


__all__ = ["interval_overlap", "panel_summary"]
