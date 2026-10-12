"""The human-labelled gold standard for citation correctness, and the judge measured
against it.

ALCE validates its automatic citation metric against human annotation and reports Cohen's
kappa - 0.698 for recall, 0.525 for precision, on 100 examples per dataset. This arm has
never done that: the judge here is calibrated on gold answers against chunks, which is a
proxy for the pairs actually scored, not a sample of them. At 16 cited sentences, labelling
every pair by hand costs less than any correction and settles both questions at once - the
true rate, and whether the judge agrees on the distribution that matters.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import pytest

from eval.citation_gold import (
    SUPPORT,
    agreement,
    cohen_kappa,
    gold_pairs,
    human_counts,
    load_labels,
)


class TestCohenKappa:
    """Chance-corrected agreement, because raw agreement flatters a skewed set. If 14 of
    16 pairs are unsupported, a judge that says "unsupported" every time is 87.5% right
    and worthless, and kappa is what says so."""

    def test_perfect_agreement_is_one(self):
        assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0

    def test_chance_level_agreement_is_about_zero(self):
        # Both raters say True half the time, and agree exactly as often as chance.
        human = [True, True, False, False]
        judge = [True, False, True, False]
        assert cohen_kappa(human, judge) == pytest.approx(0.0)

    def test_a_judge_that_always_says_the_same_thing_scores_zero_not_high(self):
        """The failure kappa exists to catch: 6 of 8 right by never deciding."""
        human = [True, True, False, False, False, False, False, False]
        judge = [False] * 8
        assert cohen_kappa(human, judge) == pytest.approx(0.0)

    def test_systematic_disagreement_is_negative(self):
        assert cohen_kappa([True, True, False, False], [False, False, True, True]) < 0

    def test_kappa_is_undefined_when_both_raters_are_constant_and_agree(self):
        """Expected agreement is 1, so the correction divides by zero. None, not 1.0:
        there is no evidence of skill in a set with no disagreement to resolve."""
        assert cohen_kappa([True, True], [True, True]) is None

    def test_mismatched_lengths_are_an_error_not_a_silent_truncation(self):
        with pytest.raises(ValueError):
            cohen_kappa([True, False], [True])

    def test_no_labels_means_no_kappa(self):
        assert cohen_kappa([], []) is None


class TestAgreement:
    """The judge scored against the human labels, on the pairs that were actually
    measured. This is the number the calibration could not provide."""

    def test_sensitivity_and_specificity_are_measured_against_the_human(self):
        human = [True, True, True, False, False, False, False]
        judge = [True, True, False, False, False, False, True]

        out = agreement(human, judge)

        assert out["n"] == 7
        assert out["sensitivity"] == pytest.approx(2 / 3)
        assert out["specificity"] == pytest.approx(3 / 4)

    def test_the_counts_are_reported_so_the_rates_can_be_checked(self):
        human = [True, True, False]
        judge = [True, False, False]

        out = agreement(human, judge)

        assert out["human_supported"] == 2
        assert out["judge_supported"] == 1
        assert out["both_supported"] == 1

    def test_agreement_carries_kappa(self):
        human = [True, False, True, False]
        judge = [True, False, True, False]
        assert agreement(human, judge)["kappa"] == 1.0


class TestGoldPairs:
    """One worksheet row per judgment a human has to make. The premise must be built the
    way the metric builds it, or the labels describe a different measurement."""

    structure: ClassVar[dict[str, Any]] = {
        "per_question": [
            {
                "question_id": "q-1",
                "n_contexts": 5,
                "answer": "Proof of loss is due in 60 days [1]. Pets are excluded [2][3].",
            }
        ]
    }
    blocks: ClassVar[dict[str, dict[int, str]]] = {"q-1": {1: "Block one text.", 2: "Block two text.", 3: "Block three."}}

    def test_one_support_pair_per_cited_sentence(self):
        pairs = gold_pairs(self.structure, self.blocks)
        support = [p for p in pairs if p["kind"] == SUPPORT]
        assert len(support) == 2

    def test_the_premise_is_the_cited_blocks_concatenated_as_the_metric_does_it(self):
        """ALCE's concat(C_i), which eval/citation_correctness._premise implements by
        joining with newlines. A worksheet showing different text would collect labels
        for a premise nobody scored."""
        pairs = gold_pairs(self.structure, self.blocks)
        second = next(
            p for p in pairs if p["kind"] == SUPPORT and p["citations"] == [2, 3]
        )
        assert second["premise"] == "Block two text.\nBlock three."

    def test_the_hypothesis_has_no_citation_markers(self):
        pairs = gold_pairs(self.structure, self.blocks)
        assert all("[" not in p["hypothesis"] for p in pairs)

    def test_single_citation_sentences_get_no_necessity_pair(self):
        """A sole citation on a supported sentence is necessary by construction - drop it
        and nothing remains to support the claim. Asking a human is wasted effort."""
        pairs = gold_pairs(self.structure, self.blocks)
        necessity = [p for p in pairs if p["kind"] != SUPPORT]
        assert all(len(p["citations"]) > 1 for p in necessity)

    def test_one_necessity_pair_per_citation_on_a_multi_citation_sentence(self):
        pairs = gold_pairs(self.structure, self.blocks)
        necessity = [p for p in pairs if p["kind"] != SUPPORT]
        assert len(necessity) == 2
        assert {p["dropped"] for p in necessity} == {2, 3}

    def test_a_necessity_premise_excludes_the_dropped_citation(self):
        pairs = gold_pairs(self.structure, self.blocks)
        dropped_two = next(p for p in pairs if p.get("dropped") == 2)
        assert dropped_two["premise"] == "Block three."
        assert "Block two text." not in dropped_two["premise"]

    def test_pair_ids_are_unique_and_stable(self):
        first = gold_pairs(self.structure, self.blocks)
        again = gold_pairs(self.structure, self.blocks)
        ids = [p["pair_id"] for p in first]
        assert len(set(ids)) == len(ids)
        assert ids == [p["pair_id"] for p in again]

    def test_pair_ids_name_the_question_so_a_label_can_be_traced(self):
        pairs = gold_pairs(self.structure, self.blocks)
        assert all(p["pair_id"].startswith("q-1") for p in pairs)


class TestLoadLabels:
    def test_labels_are_read_by_pair_id(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        path.write_text(
            '{"pair_id": "q-1#s1", "label": "attributable"}\n'
            '{"pair_id": "q-1#s2", "label": "extrapolatory"}\n',
            encoding="utf-8",
        )
        assert load_labels(path) == {"q-1#s1": ATTRIBUTABLE, "q-1#s2": EXTRAPOLATORY}

    def test_blank_lines_and_comments_are_skipped(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        path.write_text(
            '\n# a note from the annotator\n{"pair_id": "q-1#s1", "label": "attributable"}\n',
            encoding="utf-8",
        )
        assert load_labels(path) == {"q-1#s1": ATTRIBUTABLE}

    def test_an_unlabelled_file_is_empty_not_an_error(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        path.write_text("", encoding="utf-8")
        assert load_labels(path) == {}

    def test_a_missing_label_field_is_rejected(self, tmp_path):
        """An unanswered row must not read as False. Silent coercion would turn a gap in
        the labelling into evidence of an unsupported citation."""
        path = tmp_path / "labels.jsonl"
        path.write_text('{"pair_id": "q-1#s1"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="label"):
            load_labels(path)


class TestHumanCounts:
    """ALCE's recall and precision, computed from the human labels instead of a judge."""

    structure: ClassVar[dict[str, Any]] = {
        "per_question": [
            {
                "question_id": "q-1",
                "n_contexts": 5,
                "answer": "A is so [1]. B is so [2][3]. C is so [4].",
            }
        ]
    }
    blocks: ClassVar[dict[str, dict[int, str]]] = {"q-1": {n: f"Block {n}." for n in range(1, 6)}}

    def labels(self, **over):
        pairs = gold_pairs(self.structure, self.blocks)
        base = {p["pair_id"]: EXTRAPOLATORY for p in pairs}
        base.update(over)
        return pairs, base

    def test_recall_counts_supported_sentences_over_cited_sentences(self):
        pairs, labels = self.labels()
        for p in pairs:
            if p["kind"] == SUPPORT and p["citations"] in ([1], [4]):
                labels[p["pair_id"]] = ATTRIBUTABLE

        out = human_counts(pairs, labels)

        assert out["recall"] == {"supported": 2, "cited": 3}

    def test_precision_only_counts_citations_on_supported_sentences(self):
        """An unsupported sentence has no attribution to apportion. Counting its
        citations as imprecise conflates two different failures."""
        pairs, labels = self.labels()
        support = {p["citations"][0] if p["citations"] else None: p for p in pairs}
        del support
        for p in pairs:
            if p["kind"] == SUPPORT and p["citations"] == [1]:
                labels[p["pair_id"]] = ATTRIBUTABLE

        out = human_counts(pairs, labels)

        # Only the one-citation supported sentence contributes, and a sole citation is
        # necessary by construction.
        assert out["precision"] == {"needed": 1, "offered": 1}

    def test_a_citation_is_needed_when_the_rest_no_longer_support_the_sentence(self):
        pairs, labels = self.labels()
        for p in pairs:
            if p["kind"] == SUPPORT and p["citations"] == [2, 3]:
                labels[p["pair_id"]] = ATTRIBUTABLE
        # Dropping 2 leaves 3, which does NOT support -> 2 was needed.
        # Dropping 3 leaves 2, which DOES support -> 3 was not needed.
        for p in pairs:
            if p.get("dropped") == 3:
                labels[p["pair_id"]] = ATTRIBUTABLE

        out = human_counts(pairs, labels)

        assert out["precision"] == {"needed": 1, "offered": 2}

    def test_unlabelled_pairs_are_reported_rather_than_assumed(self):
        pairs = gold_pairs(self.structure, self.blocks)
        out = human_counts(pairs, {})
        assert out["unlabelled"] == len(pairs)

    def test_counts_are_none_until_every_support_pair_is_labelled(self):
        """A partial pass must not publish a rate. Half the labels give a rate over half
        the sentences, which is a different quantity wearing the same name."""
        pairs, labels = self.labels()
        labels.pop(next(p["pair_id"] for p in pairs if p["kind"] == SUPPORT))
        out = human_counts(pairs, labels)
        assert out["recall"] is None


def test_the_worksheet_pairs_match_the_pairs_the_metric_scored():
    """The guard that makes the whole exercise meaningful: if the worksheet is built from
    a different answer set than the published rates, the human labels measure something
    else. Both come from eval/citation_structure.json, and this pins the count."""
    import pathlib

    from eval.citation_gold import build_blocks

    root = pathlib.Path(__file__).resolve().parents[1]
    structure = json.loads(
        (root / "eval" / "citation_structure.json").read_text(encoding="utf-8")
    )
    correctness = json.loads(
        (root / "eval" / "citation_correctness.json").read_text(encoding="utf-8")
    )
    pairs = gold_pairs(structure, build_blocks())
    support = [p for p in pairs if p["kind"] == SUPPORT]
    assert len(support) == correctness["claims_only"]["cited_sentences"]
    assert (
        correctness["summary"]["cited_sentences"]
        - correctness["non_claims"]["excluded"]
        == len(support)
    )


class TestNullIsUnansweredNotFalse:
    """The worksheet ships pre-seeded with `"label": null` for every row, so the
    annotator edits values rather than writing JSONL by hand. `bool(None)` is False, so
    reading null as a label would silently convert every unanswered row into evidence
    that the citation is unsupported - and a half-finished pass would publish a rate."""

    def test_a_null_label_is_skipped_rather_than_read_as_false(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        path.write_text(
            '{"pair_id": "q-1#s1", "label": "attributable"}\n'
            '{"pair_id": "q-1#s2", "label": null}\n',
            encoding="utf-8",
        )
        assert load_labels(path) == {"q-1#s1": ATTRIBUTABLE}

    def test_a_fully_null_file_leaves_the_rate_unreported(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        path.write_text('{"pair_id": "q-1#s1", "label": null}\n', encoding="utf-8")
        structure: ClassVar[dict[str, Any]] = {
            "per_question": [
                {"question_id": "q-1", "n_contexts": 5, "answer": "A is so [1]."}
            ]
        }
        pairs = gold_pairs(structure, {"q-1": {1: "Block one."}})
        assert human_counts(pairs, load_labels(path))["recall"] is None


class TestJudgeVerdictsPerSentence:
    """Agreement must be measurable on every labelled pair, not only the convenient ones.

    Per-question rates cannot be unpicked: a question with 3 cited sentences and a recall
    of 0.333 says one of them was supported, not which. Recovering verdicts from rates
    would silently drop the multi-sentence questions - 5 of 16 pairs here - from the only
    comparison that validates the judge."""

    structure: ClassVar[dict[str, Any]] = {
        "per_question": [
            {
                "question_id": "q-1",
                "n_contexts": 5,
                "answer": "A is so [1]. B is so [2]. C is so [3].",
            }
        ]
    }
    blocks: ClassVar[dict[str, dict[int, str]]] = {"q-1": {1: "Block one.", 2: "Block two.", 3: "Block three."}}

    def test_explicit_per_sentence_decisions_are_used_when_present(self):
        from eval.citation_gold_run import _judge_support

        correctness = {
            "per_question": [
                {
                    "question_id": "q-1",
                    "cited_sentences": 3,
                    "recall": 1 / 3,
                    "sentences": [
                        {"index": 1, "supported": False},
                        {"index": 2, "supported": True},
                        {"index": 3, "supported": False},
                    ],
                }
            ]
        }
        pairs = gold_pairs(self.structure, self.blocks)

        out = _judge_support(correctness, pairs)

        assert out == {"q-1#s1": False, "q-1#s2": True, "q-1#s3": False}

    def test_a_multi_sentence_question_without_decisions_is_omitted_not_guessed(self):
        from eval.citation_gold_run import _judge_support

        correctness = {
            "per_question": [
                {"question_id": "q-1", "cited_sentences": 3, "recall": 1 / 3}
            ]
        }
        pairs = gold_pairs(self.structure, self.blocks)

        assert _judge_support(correctness, pairs) == {}


class TestNonClaimsAreNotPutToAHuman:
    """Only a hypothesis with nothing left in it is withheld from the annotator.

    An intermediate version also withheld anything starting with a conjunction, which
    discarded a real claim - "and According to the context, ... the revised due date will
    be 30 days after the date on which the bill is mailed". Fragments go to the person;
    deciding whether "provide relevant information" is supported is a judgment about
    meaning, and a regex making it is what went wrong."""

    structure: ClassVar[dict[str, Any]] = {
        "per_question": [
            {
                "question_id": "q-1",
                "n_contexts": 5,
                "answer": "[1] [2] and provide relevant information. "
                "The limit is $30,000 [3]. [4]",
            }
        ]
    }
    blocks: ClassVar[dict[str, dict[int, str]]] = {
        "q-1": {1: "One.", 2: "Two.", 3: "Three.", 4: "Four."}
    }

    def test_a_sentence_that_was_only_citations_gets_no_worksheet_row(self):
        pairs = gold_pairs(self.structure, self.blocks)
        hypotheses = [p["hypothesis"] for p in pairs if p["kind"] == SUPPORT]
        assert "" not in hypotheses

    def test_a_fragment_is_still_offered_for_labelling(self):
        pairs = gold_pairs(self.structure, self.blocks)
        hypotheses = [p["hypothesis"] for p in pairs if p["kind"] == SUPPORT]
        assert "provide relevant information." in hypotheses
        assert "The limit is $30,000." in hypotheses

    def test_only_the_empty_one_is_excluded_and_it_is_counted(self):
        from eval.citation_gold import excluded_non_claims

        out = excluded_non_claims(self.structure)
        assert out["excluded"] == 1
        assert out["of"] == 3
        assert out["examples"] == ["(empty)"]


# -------------------------------------- AttrScore's three-way label, not a binary one

from eval.citation_gold import ATTRIBUTABLE, CONTRADICTORY, EXTRAPOLATORY, is_supported


class TestThreeWayLabel:
    """AttrScore's categories (Yue et al., EMNLP Findings 2023, arXiv:2305.06311):
    attributable, extrapolatory, contradictory. Binary conflates the last two, and they
    are different failures - "the blocks do not mention the time frame" is a retrieval or
    coverage problem, "the blocks say otherwise" is a generation problem. Several
    sentences in this corpus are plainly extrapolatory, so the distinction costs the
    annotator nothing and the binary throws it away.

    Recall and precision are unchanged: attributable is supported, the other two are not.
    """

    def test_attributable_is_supported(self):
        assert is_supported(ATTRIBUTABLE) is True

    def test_extrapolatory_is_not_supported(self):
        assert is_supported(EXTRAPOLATORY) is False

    def test_contradictory_is_not_supported(self):
        assert is_supported(CONTRADICTORY) is False

    def test_an_unanswered_label_is_not_supported_and_not_a_label(self):
        assert is_supported(None) is False

    def test_three_way_labels_are_read(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        path.write_text(
            '{"pair_id": "q-1#s1", "label": "attributable"}\n'
            '{"pair_id": "q-1#s2", "label": "extrapolatory"}\n'
            '{"pair_id": "q-1#s3", "label": "contradictory"}\n',
            encoding="utf-8",
        )
        assert load_labels(path) == {
            "q-1#s1": ATTRIBUTABLE,
            "q-1#s2": EXTRAPOLATORY,
            "q-1#s3": CONTRADICTORY,
        }

    def test_a_null_label_is_still_unanswered(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        path.write_text('{"pair_id": "q-1#s1", "label": null}\n', encoding="utf-8")
        assert load_labels(path) == {}

    def test_a_misspelled_label_is_rejected_rather_than_guessed(self, tmp_path):
        """A typo must not silently become "not supported". That would turn an
        annotator's slip into evidence against the system."""
        path = tmp_path / "labels.jsonl"
        path.write_text(
            '{"pair_id": "q-1#s1", "label": "attributible"}\n', encoding="utf-8"
        )
        with pytest.raises(ValueError, match="attributible"):
            load_labels(path)

    def test_a_row_with_no_label_field_is_rejected(self, tmp_path):
        path = tmp_path / "labels.jsonl"
        path.write_text('{"pair_id": "q-1#s1"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="label"):
            load_labels(path)


class TestThreeWayFeedsTheSameRates:
    structure: ClassVar[dict[str, Any]] = {
        "per_question": [
            {
                "question_id": "q-1",
                "n_contexts": 5,
                "answer": "A is so [1]. B is so [2]. C is so [3].",
            }
        ]
    }
    blocks: ClassVar[dict[str, dict[int, str]]] = {
        "q-1": {1: "One.", 2: "Two.", 3: "Three."}
    }

    def test_only_attributable_counts_towards_recall(self):
        pairs = gold_pairs(self.structure, self.blocks)
        labels = {
            "q-1#s1": ATTRIBUTABLE,
            "q-1#s2": EXTRAPOLATORY,
            "q-1#s3": CONTRADICTORY,
        }
        assert human_counts(pairs, labels)["recall"] == {"supported": 1, "cited": 3}

    def test_the_failure_breakdown_is_reported(self):
        """The reason for three labels instead of two: which kind of failure, and how
        often. A rate of 1/3 does not say whether the evidence was missing or wrong."""
        pairs = gold_pairs(self.structure, self.blocks)
        labels = {
            "q-1#s1": ATTRIBUTABLE,
            "q-1#s2": EXTRAPOLATORY,
            "q-1#s3": CONTRADICTORY,
        }
        out = human_counts(pairs, labels)
        assert out["breakdown"] == {
            ATTRIBUTABLE: 1,
            EXTRAPOLATORY: 1,
            CONTRADICTORY: 1,
        }


class TestDisagreementFirstOrdering:
    """Where two calibrated judges disagree is where a label is worth most.

    Both judges clear the declared floors and differ on recall by a third, and the five
    sentences they split on are the only ones that can settle which to believe. Agreed
    pairs mostly confirm what is already known. This is the cheap version of the
    epistemic-uncertainty targeting in arXiv:2609.06444 - whose authors note that simple
    heuristics performed comparably - and at 36 judgments it only changes the order, not
    what gets labelled, so nothing is lost if the ordering is wrong.
    """

    def test_contested_pairs_come_first(self):
        from eval.citation_gold import order_by_information

        pairs = [
            {"pair_id": "a#s1", "kind": SUPPORT},
            {"pair_id": "b#s1", "kind": SUPPORT},
            {"pair_id": "c#s1", "kind": SUPPORT},
        ]
        out = order_by_information(pairs, contested={"c#s1"})
        assert next(p["pair_id"] for p in out) == "c#s1"

    def test_every_pair_survives_the_reordering(self):
        from eval.citation_gold import order_by_information

        pairs = [{"pair_id": f"q#s{i}", "kind": SUPPORT} for i in range(5)]
        out = order_by_information(pairs, contested={"q#s3"})
        assert sorted(p["pair_id"] for p in out) == sorted(
            p["pair_id"] for p in pairs
        )

    def test_order_is_stable_within_a_group(self):
        """Two runs must produce the same worksheet, or a half-finished labelling pass
        cannot be resumed against it."""
        from eval.citation_gold import order_by_information

        pairs = [{"pair_id": f"q#s{i}", "kind": SUPPORT} for i in range(6)]
        contested = {"q#s4", "q#s1"}
        first = [p["pair_id"] for p in order_by_information(pairs, contested)]
        again = [p["pair_id"] for p in order_by_information(pairs, contested)]
        assert first == again
        assert first[:2] == ["q#s1", "q#s4"]

    def test_no_contested_set_leaves_the_order_alone(self):
        from eval.citation_gold import order_by_information

        pairs = [{"pair_id": f"q#s{i}", "kind": SUPPORT} for i in range(4)]
        out = order_by_information(pairs, contested=set())
        assert [p["pair_id"] for p in out] == [p["pair_id"] for p in pairs]

    def test_judges_disagreeing_is_read_from_the_artefacts(self):
        from eval.citation_gold import contested_pairs

        base = {
            "per_question": [
                {
                    "question_id": "q-1",
                    "sentences": [
                        {"index": 1, "supported": True},
                        {"index": 2, "supported": False},
                    ],
                }
            ]
        }
        large = {
            "per_question": [
                {
                    "question_id": "q-1",
                    "sentences": [
                        {"index": 1, "supported": True},
                        {"index": 2, "supported": True},
                    ],
                }
            ]
        }
        assert contested_pairs(base, large) == {"q-1#s2"}


class TestBuildDoesNotDestroyAnswers:
    """`--build` creates the labels file, so it must not crash on one it cannot read,
    and must never overwrite one that might hold answers.

    This surfaced for real: changing the schema from a binary `supported` to AttrScore's
    three-way `label` made every existing row unreadable, and `--build` raised a
    traceback from inside the function whose job is to produce that file."""

    def test_an_unreadable_labels_file_is_reported_not_crashed(self, tmp_path):
        from eval.citation_gold import answered_count

        path = tmp_path / "labels.jsonl"
        path.write_text('{"pair_id": "q-1#s1", "supported": null}\n', encoding="utf-8")

        # Unreadable under the current schema; the caller needs a verdict, not an
        # exception, so it can decide whether overwriting is safe.
        assert answered_count(path) is None

    def test_a_readable_file_reports_how_many_are_answered(self, tmp_path):
        from eval.citation_gold import answered_count

        path = tmp_path / "labels.jsonl"
        path.write_text(
            '{"pair_id": "q-1#s1", "label": "attributable"}\n'
            '{"pair_id": "q-1#s2", "label": null}\n',
            encoding="utf-8",
        )
        assert answered_count(path) == 1

    def test_a_missing_file_has_nothing_to_lose(self, tmp_path):
        from eval.citation_gold import answered_count

        assert answered_count(tmp_path / "nope.jsonl") == 0
