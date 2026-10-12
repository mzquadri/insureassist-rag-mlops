"""Write the judge-panel artefact from the two correctness runs.

    python -m eval.judge_panel_run

Offline and deterministic: no model is loaded and nothing is re-scored. Both correctness
artefacts already hold per-sentence verdicts, which is what makes the paired comparison
possible at all - a rate cannot be unpicked into which sentence it disagreed about.

Writes `eval/judge_panel.json`. See `eval/judge_panel.py` for why the spread is reported
instead of a winner being chosen.
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

from eval.judge_panel import interval_overlap, panel_summary

OUT = ROOT / "eval" / "judge_panel.json"
ARTEFACTS = {
    "base": ROOT / "eval" / "citation_correctness.json",
    "large": ROOT / "eval" / "citation_correctness_large.json",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, as in scripts/figures/generate_figures.py."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def main() -> int:
    missing = [str(p.relative_to(ROOT)) for p in ARTEFACTS.values() if not p.exists()]
    if missing:
        print(f"missing: {', '.join(missing)}")
        print("Run eval/citation_correctness_run.py --judge base|large first.")
        return 1

    artefacts = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in ARTEFACTS.items()
    }
    summary = panel_summary(artefacts)

    intervals = {}
    for name, a in artefacts.items():
        counts = a["claims_only"]["recall_counts"]
        low, high = wilson(counts["supported"], counts["cited"])
        intervals[name] = [round(low, 3), round(high, 3)]
        summary["judges"][name]["recall_wilson_95"] = intervals[name]

    overlap = interval_overlap(tuple(intervals["base"]), tuple(intervals["large"]))

    payload: dict[str, Any] = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "measures": (
            "How far the citation-correctness result depends on which entailment judge "
            "is used, rather than on the system being measured."
        ),
        "why": (
            "Both judges clear sensitivity and specificity floors declared before either "
            "was run, and disagree on recall by more than the sampling interval around "
            "either figure. AttributionBench (arXiv:2402.15089) finds no single automatic "
            "evaluator suffices, and that evaluators are weakest on long evidence in "
            "expert domains where claims turn on numbers and dates - which is this corpus."
        ),
        "no_winner_is_chosen": (
            "Only the 0.80/0.80 floors were declared in advance. Selecting on balanced "
            "accuracy after seeing which judge gives the friendlier recall is the "
            "post-hoc choice the floors exist to prevent. eval/citation_gold.py "
            "adjudicates, with human labels."
        ),
        **summary,
        "recall_interval_overlap": round(overlap, 3),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for name, j in summary["judges"].items():
        c, cl = j["calibration"], j["claims_only"]
        print(
            f"{name:6s} cal sens {c['sensitivity']} spec {c['specificity']}   "
            f"recall {cl['citation_recall']:.4f} {j['recall_wilson_95']}   "
            f"precision {cl['citation_precision']:.4f}"
        )
    p = summary["paired"]
    print(
        f"\nrecall spread {summary['recall_spread']:.4f}   "
        f"interval overlap {overlap:.3f}"
    )
    print(
        f"paired n={p['sentences']} both={p['both']} neither={p['neither']} "
        f"only_base={p['only_base']} only_large={p['only_large']} "
        f"nested={p['nested']} p={p['p_mcnemar_exact']}"
    )
    print(f"contested: {', '.join(p['contested'])}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
