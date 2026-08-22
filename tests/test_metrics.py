"""Retrieval metrics, checked against hand-computed answers."""
import math

import pytest

from eval.metrics import (
    deduplicate,
    hit_rate_at_k,
    mean,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["a", "b", "c", "d", "e"]


class TestDeduplicate:
    def test_keeps_first_occurrence_and_order(self):
        assert deduplicate(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_leaves_clean_input_untouched(self):
        assert deduplicate(RANKED) == RANKED

    def test_empty(self):
        assert deduplicate([]) == []


class TestHitRate:
    def test_relevant_at_rank_one(self):
        assert hit_rate_at_k(RANKED, {"a"}, 1) == 1.0

    def test_relevant_below_the_cutoff(self):
        assert hit_rate_at_k(RANKED, {"d"}, 3) == 0.0
        assert hit_rate_at_k(RANKED, {"d"}, 5) == 1.0

    def test_any_relevant_counts(self):
        assert hit_rate_at_k(RANKED, {"c", "z"}, 3) == 1.0

    def test_nothing_relevant_retrieved(self):
        assert hit_rate_at_k(RANKED, {"z"}, 5) == 0.0

    def test_empty_relevance_is_nan_not_zero(self):
        """An unanswerable question has no recall to measure; zero would invent a failure."""
        assert math.isnan(hit_rate_at_k(RANKED, set(), 5))

    def test_duplicates_do_not_shift_ranks(self):
        assert hit_rate_at_k(["a", "a", "d"], {"d"}, 2) == 1.0


class TestRecall:
    def test_single_relevant_found(self):
        assert recall_at_k(RANKED, {"a"}, 1) == 1.0

    def test_partial_multi_chunk(self):
        assert recall_at_k(RANKED, {"a", "z"}, 5) == pytest.approx(0.5)

    def test_all_relevant_found(self):
        assert recall_at_k(RANKED, {"a", "b"}, 3) == 1.0

    def test_cutoff_excludes_one(self):
        assert recall_at_k(RANKED, {"a", "e"}, 3) == pytest.approx(0.5)

    def test_none_found(self):
        assert recall_at_k(RANKED, {"y", "z"}, 5) == 0.0

    def test_empty_relevance_is_nan(self):
        assert math.isnan(recall_at_k(RANKED, set(), 5))

    def test_duplicates_are_not_double_counted(self):
        assert recall_at_k(["a", "a", "a"], {"a", "b"}, 5) == pytest.approx(0.5)


class TestPrecision:
    def test_one_relevant_in_five(self):
        assert precision_at_k(RANKED, {"a"}, 5) == pytest.approx(0.2)

    def test_two_relevant_in_five(self):
        assert precision_at_k(RANKED, {"a", "b"}, 5) == pytest.approx(0.4)

    def test_perfect_at_one(self):
        assert precision_at_k(RANKED, {"a"}, 1) == 1.0

    def test_bounded_by_relevant_over_k(self):
        """With one relevant chunk, precision@5 can never exceed 0.2."""
        assert precision_at_k(RANKED, {"c"}, 5) <= 0.2

    def test_empty_retrieval(self):
        assert precision_at_k([], {"a"}, 5) == 0.0

    def test_empty_relevance_is_nan(self):
        assert math.isnan(precision_at_k(RANKED, set(), 5))

    def test_duplicates_do_not_inflate(self):
        assert precision_at_k(["a", "a"], {"a"}, 2) == pytest.approx(1.0)


class TestReciprocalRank:
    @pytest.mark.parametrize("relevant,expected", [
        ({"a"}, 1.0), ({"b"}, 0.5), ({"c"}, 1 / 3), ({"d"}, 0.25), ({"e"}, 0.2),
    ])
    def test_rank_positions(self, relevant, expected):
        assert reciprocal_rank(RANKED, relevant) == pytest.approx(expected)

    def test_first_relevant_wins(self):
        assert reciprocal_rank(RANKED, {"b", "d"}) == pytest.approx(0.5)

    def test_not_retrieved(self):
        assert reciprocal_rank(RANKED, {"z"}) == 0.0

    def test_empty_relevance_is_nan(self):
        assert math.isnan(reciprocal_rank(RANKED, set()))

    def test_duplicates_do_not_change_the_rank(self):
        assert reciprocal_rank(["a", "a", "b"], {"b"}) == pytest.approx(0.5)

    def test_mean_reciprocal_rank(self):
        results = [(RANKED, {"a"}), (RANKED, {"b"}), (RANKED, {"z"})]
        assert mean_reciprocal_rank(results) == pytest.approx((1.0 + 0.5 + 0.0) / 3)

    def test_mrr_skips_unanswerable(self):
        results = [(RANKED, {"a"}), (RANKED, set())]
        assert mean_reciprocal_rank(results) == pytest.approx(1.0)


class TestMean:
    def test_ignores_nan(self):
        assert mean([1.0, float("nan"), 0.0]) == pytest.approx(0.5)

    def test_all_nan_is_nan(self):
        assert math.isnan(mean([float("nan"), float("nan")]))

    def test_empty_is_nan(self):
        assert math.isnan(mean([]))


class TestNdcgIsDeferred:
    def test_ndcg_is_not_exported(self):
        """Labels are binary, so nDCG would add a column without adding information."""
        from eval import metrics

        assert not hasattr(metrics, "ndcg_at_k")
