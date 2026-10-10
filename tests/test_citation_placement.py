"""Where the generator puts its citation markers, as a fact about the string.

This arm exists because of a defect it explains. Citation correctness was reported from
hypotheses that were malformed: the model emits `"[2], [5]\\n\\nThe maximum payable ..."`,
so stripping the markers left punctuation glued to the front of nine of sixteen sentences.
Fixing the extraction moved recall from 0.3125 to 0.4667.

That raised the question the fix could not answer: is the placement incidental, or is it how
this model cites? No judge is involved here - whether a sentence begins with a marker is
decidable by reading the characters, which is the same reason the dangling-citation rate in
`eval/citation_structure.py` survives the circularity objection that blocks answer-quality
scoring in this repository.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from eval.citation_placement import (
    coordinates_citations,
    leads_with_citation,
    mcnemar_exact,
    placement_counts,
)


class TestLeadsWithCitation:
    """A citation that precedes its prose is not attached to a claim.

    ALCE's metric, and the `[n]` convention generally, assume the marker follows the
    statement it supports. A sentence that opens with one leaves which claim it supports
    ambiguous, and it is what broke the hypothesis extraction.
    """

    def test_a_sentence_opening_with_a_marker_leads(self):
        assert leads_with_citation("[2] The maximum payable is $30,000.") is True

    def test_several_markers_before_the_prose_still_lead(self):
        assert leads_with_citation("[2], [5] The maximum payable is $30,000.") is True

    def test_leading_whitespace_does_not_hide_it(self):
        assert leads_with_citation("   [3] Proof of loss is due.") is True

    def test_a_trailing_citation_does_not_lead(self):
        assert leads_with_citation("The maximum payable is $30,000. [2]") is False

    def test_a_mid_sentence_citation_does_not_lead(self):
        assert leads_with_citation("According to [2], the Dwelling Form says so.") is (
            False
        )

    def test_a_sentence_with_no_citation_does_not_lead(self):
        assert leads_with_citation("The maximum payable is $30,000.") is False

    def test_a_bracket_that_is_not_a_citation_does_not_count(self):
        assert leads_with_citation("[see below] the limit applies.") is False


class TestCoordinatesCitations:
    """Markers joined by a conjunction are being used as a noun phrase - "[2], [3], and
    [4] provide relevant information" - rather than as an attachment. Counted separately
    from leading, because a sentence can do either, both, or neither."""

    def test_markers_joined_by_and_are_coordinated(self):
        assert coordinates_citations("[2], [3], and [4] provide information.") is True

    def test_two_markers_joined_by_and_are_coordinated(self):
        assert coordinates_citations("However, [3] and [4] do not say.") is True

    def test_markers_joined_by_or_are_coordinated(self):
        assert coordinates_citations("[1] or [2] mention the deadline.") is True

    def test_adjacent_markers_without_a_conjunction_are_not_coordinated(self):
        assert coordinates_citations("[2], [5] The maximum payable is $30,000.") is False

    def test_a_single_marker_is_not_coordinated(self):
        assert coordinates_citations("[2] The maximum payable is $30,000.") is False

    def test_prose_containing_and_between_words_is_not_coordinated(self):
        assert coordinates_citations("Coverage A and Coverage D differ. [1]") is False


class TestMcnemarExact:
    """Two-sided exact McNemar, for paired yes/no outcomes on the same questions.

    The repository already quotes p = 0.250 for the prompt comparison in
    docs/LIMITATIONS.md, computed by hand. This makes that number reproducible, and the
    first test pins it.
    """

    def test_three_discordant_pairs_all_one_way_give_the_published_p(self):
        assert mcnemar_exact(3, 0) == pytest.approx(0.250)

    def test_no_discordant_pairs_cannot_distinguish_anything(self):
        assert mcnemar_exact(0, 0) == 1.0

    def test_one_discordant_pair_is_no_evidence(self):
        assert mcnemar_exact(1, 0) == 1.0

    def test_the_test_is_symmetric_in_its_arguments(self):
        assert mcnemar_exact(5, 2) == mcnemar_exact(2, 5)

    def test_an_even_split_is_maximally_inconclusive(self):
        assert mcnemar_exact(4, 4) == 1.0

    def test_a_large_lopsided_split_is_significant(self):
        assert mcnemar_exact(10, 0) < 0.01

    def test_the_result_is_a_probability(self):
        for b in range(8):
            for c in range(8):
                assert 0.0 <= mcnemar_exact(b, c) <= 1.0


class TestPlacementCounts:
    """Counts over one artefact, per question and in total."""

    structure: ClassVar[dict[str, Any]] = {
        "per_question": [
            {
                "question_id": "q-1",
                "n_contexts": 5,
                "answer": "[2] The limit is $30,000.",
            },
            {
                "question_id": "q-2",
                "n_contexts": 5,
                "answer": "The limit is $30,000 [2].",
            },
            {"question_id": "q-3", "n_contexts": 5, "answer": "No citation here."},
        ]
    }

    def test_totals_count_cited_sentences_not_questions(self):
        out = placement_counts(self.structure)
        assert out["cited_sentences"] == 2
        assert out["leading"] == 1

    def test_a_question_with_no_cited_sentence_is_recorded_as_such(self):
        out = placement_counts(self.structure)
        assert out["per_question"]["q-3"]["cited_sentences"] == 0

    def test_per_question_flags_whether_any_sentence_leads(self):
        out = placement_counts(self.structure)
        assert out["per_question"]["q-1"]["any_leading"] is True
        assert out["per_question"]["q-2"]["any_leading"] is False

    def test_a_question_that_cited_nothing_has_no_placement_verdict(self):
        """Absence of citations is not good placement. A question with nothing cited must
        not be paired against one that cited well, which is the confound that makes the
        raw served-versus-ranged difference misleading."""
        out = placement_counts(self.structure)
        assert out["per_question"]["q-3"]["any_leading"] is None
