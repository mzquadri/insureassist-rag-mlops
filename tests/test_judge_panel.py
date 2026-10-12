"""The spread between judges, as data rather than prose.

Two judges clear the same declared calibration floors and differ on citation recall by a
third. That is the dominant uncertainty in this arm - larger than the sampling interval
around either figure - and until now it lived only in a paragraph of `docs/LIMITATIONS.md`.
Prose drifts; a pinned artefact does not.

The practice being followed here is the one the attribution-evaluation literature settled
on: score the same outputs with a panel and report the instrument dependence, rather than
hunting for a single better judge. AttributionBench (arXiv:2402.15089) finds no single
automatic evaluator suffices, and that evaluators are weakest on exactly this shape of
input - long evidence, expert domain, claims turning on numbers and dates.
"""

from __future__ import annotations

from typing import Any

import pytest

from eval.judge_panel import interval_overlap, panel_summary


def artefact(sentences, sensitivity=0.9, recall=0.5, precision=0.4):
    """A correctness artefact with just the fields the panel reads."""
    return {
        "judge": {"name": "x", "model": "m"},
        "calibration": {
            "sensitivity": sensitivity,
            "specificity": 0.96,
            "threshold": 0.5,
            "usable": True,
        },
        "claims_only": {
            "citation_recall": recall,
            "recall_counts": {"supported": 7, "cited": 15},
            "citation_precision": precision,
            "precision_counts": {"needed": 5, "offered": 15},
        },
        "per_question": [
            {
                "question_id": "q-1",
                "sentences": [
                    {"index": i, "supported": v} for i, v in enumerate(sentences, 1)
                ],
            }
        ],
    }


class TestIntervalOverlap:
    """How much of each judge's interval the other covers. Zero overlap would mean the
    two cannot both be right; heavy overlap would mean the disagreement is sampling
    noise. Neither is the case here, which is why it is worth a number."""

    def test_disjoint_intervals_do_not_overlap(self):
        assert interval_overlap((0.1, 0.3), (0.5, 0.9)) == 0.0

    def test_touching_intervals_do_not_overlap(self):
        assert interval_overlap((0.1, 0.5), (0.5, 0.9)) == 0.0

    def test_partial_overlap_is_its_width(self):
        assert interval_overlap((0.248, 0.699), (0.548, 0.930)) == pytest.approx(0.151)

    def test_a_contained_interval_overlaps_by_its_own_width(self):
        assert interval_overlap((0.2, 0.4), (0.1, 0.9)) == pytest.approx(0.2)

    def test_order_does_not_matter(self):
        assert interval_overlap((0.5, 0.9), (0.1, 0.3)) == interval_overlap(
            (0.1, 0.3), (0.5, 0.9)
        )


class TestPanelSummary:
    def test_the_contested_sentences_are_named(self):
        base = artefact([True, False, False])
        large = artefact([True, True, False])

        out = panel_summary({"base": base, "large": large})

        assert out["paired"]["contested"] == ["q-1#s2"]

    def test_the_contingency_is_reported(self):
        base = artefact([True, False, False])
        large = artefact([True, True, False])

        out = panel_summary({"base": base, "large": large})["paired"]

        assert out["both"] == 1
        assert out["neither"] == 1
        assert out["only_large"] == 1
        assert out["only_base"] == 0

    def test_a_one_directional_split_is_flagged(self):
        """One judge strictly stricter than the other is a different situation from two
        judges making independent mistakes, and calls for a different remedy."""
        base = artefact([True, False, False])
        large = artefact([True, True, True])

        assert panel_summary({"base": base, "large": large})["paired"]["nested"] is True

    def test_judges_that_each_win_some_are_not_nested(self):
        base = artefact([True, False])
        large = artefact([False, True])

        assert panel_summary({"base": base, "large": large})["paired"]["nested"] is False

    def test_mcnemar_is_computed_on_the_discordant_pairs(self):
        base = artefact([True, False, False, False, False, False])
        large = artefact([True, True, True, True, True, True])

        out = panel_summary({"base": base, "large": large})["paired"]

        # five discordant pairs, all one way
        assert out["only_large"] == 5
        assert out["p_mcnemar_exact"] == pytest.approx(0.0625)

    def test_the_recall_spread_is_the_headline(self):
        base = artefact([True], recall=0.4667)
        large = artefact([True], recall=0.8000)

        out = panel_summary({"base": base, "large": large})

        assert out["recall_spread"] == pytest.approx(0.3333, abs=1e-4)

    def test_each_judge_keeps_its_calibration_beside_its_rate(self):
        """A rate is unreadable without the calibration that licensed it, so the panel
        carries both or it is just two numbers."""
        base = artefact([True], sensitivity=0.8438)
        large = artefact([True], sensitivity=0.9375)

        out = panel_summary({"base": base, "large": large})

        assert out["judges"]["base"]["calibration"]["sensitivity"] == 0.8438
        assert out["judges"]["large"]["calibration"]["sensitivity"] == 0.9375

    def test_a_single_judge_has_no_panel_to_report(self):
        out: dict[str, Any] = panel_summary({"base": artefact([True])})
        assert out["paired"] is None
