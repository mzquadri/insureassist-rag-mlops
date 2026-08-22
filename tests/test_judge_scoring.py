"""
Judge parsing and NaN accounting.

Both are pure functions, so the rules can be pinned without a model. That matters: a small
judge frequently returns something other than a number, and the old code silently dropped
those rows from the average without saying so.
"""
import math

import pytest

from eval.evaluate import METRICS, average_scores, parse_judge_score


class TestParseJudgeScore:
    @pytest.mark.parametrize("text,expected", [
        ("0.8", 0.8),
        ("1.0", 1.0),
        ("0", 0.0),
        ("1", 1.0),
        ("0.85", 0.85),
    ])
    def test_plain_numbers(self, text, expected):
        assert parse_judge_score(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text,expected", [
        ("Score: 0.7", 0.7),
        ("  0.6  ", 0.6),
        ("0.9\n\nBecause the answer is supported.", 0.9),
    ])
    def test_numbers_embedded_in_prose(self, text, expected):
        """Small judges pad their answer despite being told not to."""
        assert parse_judge_score(text) == pytest.approx(expected)

    @pytest.mark.parametrize("text", ["", "   ", "I cannot score this.", "N/A", "high"])
    def test_unparseable_replies_become_nan(self, text):
        assert math.isnan(parse_judge_score(text))

    def test_none_becomes_nan(self):
        assert math.isnan(parse_judge_score(None))

    def test_values_are_clamped_to_the_unit_interval(self):
        assert parse_judge_score("1.0") == 1.0
        assert parse_judge_score("0.0") == 0.0

    def test_a_wrong_number_is_never_invented(self):
        """Refusals must degrade to NaN, never to a plausible-looking score."""
        assert math.isnan(parse_judge_score("I refuse to answer"))


class TestAverageScores:
    def _rows(self, values):
        return [{m: v for m in METRICS} for v in values]

    def test_mean_of_complete_rows(self):
        summary = average_scores(self._rows([0.8, 0.6, 1.0]), METRICS)
        assert summary["judge_faithfulness"]["mean"] == pytest.approx(0.8)
        assert summary["judge_faithfulness"]["used"] == 3
        assert summary["judge_faithfulness"]["dropped"] == 0

    def test_nan_rows_are_excluded_from_the_mean(self):
        summary = average_scores(self._rows([1.0, float("nan"), 0.0]), METRICS)
        assert summary["judge_faithfulness"]["mean"] == pytest.approx(0.5)

    def test_dropped_rows_are_counted(self):
        """The whole point: a partially-failed run must be visible.

        An average over two questions used to be indistinguishable from an average over
        ten, because the denominator was never reported.
        """
        summary = average_scores(self._rows([0.8, float("nan"), float("nan")]), METRICS)
        assert summary["judge_faithfulness"]["used"] == 1
        assert summary["judge_faithfulness"]["dropped"] == 2

    def test_all_nan_yields_nan_not_a_crash(self):
        summary = average_scores(self._rows([float("nan")] * 3), METRICS)
        assert math.isnan(summary["judge_faithfulness"]["mean"])
        assert summary["judge_faithfulness"]["used"] == 0

    def test_empty_input(self):
        summary = average_scores([], METRICS)
        assert math.isnan(summary["judge_faithfulness"]["mean"])
        assert summary["judge_faithfulness"]["dropped"] == 0

    def test_every_metric_is_summarised(self):
        summary = average_scores(self._rows([0.5]), METRICS)
        assert set(summary) == set(METRICS)


class TestRecordedReportStillReconciles:
    """The published averages must be recomputable from the committed CSV.

    `eval/README.md` prints three averages. If the CSV is ever edited, regenerated, or
    relabelled, this fails rather than letting the documented numbers drift away from the
    evidence behind them.
    """

    def test_committed_report_matches_published_averages(self):
        import csv

        with open("eval/eval_report.csv", encoding="utf-8") as f:
            rows = [{k: (float(v) if k.startswith("judge_") else v)
                     for k, v in row.items()} for row in csv.DictReader(f)]

        assert len(rows) == 10
        summary = average_scores(rows, METRICS)
        assert summary["judge_faithfulness"]["mean"] == pytest.approx(0.54)
        assert summary["judge_answer_relevancy"]["mean"] == pytest.approx(0.52)
        assert summary["judge_answer_correctness"]["mean"] == pytest.approx(0.60)

    def test_context_precision_is_degenerate(self):
        """Pins the reason context_precision was withdrawn from the headline table.

        It scored an identical 0.80 on all ten questions. That is a metric carrying no
        information, not a retriever behaving consistently, and the documentation says so.
        If this ever stops being constant the claim needs rewriting.
        """
        import csv

        with open("eval/eval_report.csv", encoding="utf-8") as f:
            values = [float(r["judge_context_precision"]) for r in csv.DictReader(f)]
        assert len(set(values)) == 1, "context_precision now varies; update eval/README.md"
        assert values[0] == pytest.approx(0.80)
