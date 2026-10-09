"""ALCE-style citation recall and precision, tested without an entailment model.

The metric definitions are Gao et al.'s (arXiv:2305.14627): recall asks whether the
blocks a sentence cites, taken together, entail that sentence; precision asks whether
each individual citation was needed for that.

Entailment arrives as a callable so the arithmetic can be exercised against a fake whose
answers are stated rather than inferred. The real judge is a separate concern and has its
own control in the runner.
"""

from __future__ import annotations

from eval.citation_correctness import (
    Claim,
    citation_precision,
    citation_precision_counts,
    citation_recall,
    citation_recall_counts,
    claims,
    hypothesis_of,
    is_claim,
    split_sentences,
)


def test_sentences_split_on_terminators():
    text = "Proof of loss is due in 60 days. The limit is $30,000."

    assert split_sentences(text) == [
        "Proof of loss is due in 60 days.",
        "The limit is $30,000.",
    ]


def test_a_decimal_does_not_end_a_sentence():
    """Policy text is full of amounts; splitting inside one would cut a claim in half."""
    assert split_sentences("The limit is $30,000.00 per occurrence.") == [
        "The limit is $30,000.00 per occurrence."
    ]


def test_a_sentence_opening_with_a_citation_is_not_merged_into_the_previous_one():
    text = "Coverage applies. [2] The deductible is separate."

    assert len(split_sentences(text)) == 2


def test_only_sentences_carrying_citations_become_claims():
    """ALCE scores cited statements. An uncited sentence has no citation to be right or
    wrong about, and counting it would mix abstention into a citation metric."""
    text = (
        "Flood damage is covered [1]. Something else entirely. The limit applies [2]."
    )

    assert [c.citations for c in claims(text, n_contexts=3)] == [[1], [2]]


def test_citations_outside_the_supplied_range_are_dropped_from_claims():
    # They are already counted as structural failures; scoring them here would punish
    # the same mistake twice and make recall unreadable.
    text = "The limit applies [9]."

    assert claims(text, n_contexts=3) == []


BLOCKS = {
    1: "Proof of loss is due in 60 days.",
    2: "The limit is $30,000.",
    3: "Unrelated.",
}


def entails_if_substring(premise: str, hypothesis: str) -> bool:
    """A stand-in judge with no opinions: entailment is literal containment."""
    return hypothesis.strip(" .") in premise


def test_recall_is_the_share_of_cited_sentences_their_citations_support():
    text = "Proof of loss is due in 60 days [1]. The limit is $1 [2]."

    assert citation_recall(claims(text, 3), BLOCKS, entails_if_substring) == 0.5


def test_recall_of_nothing_is_not_a_number_pretending_to_be_zero():
    # No cited sentences means no recall to report. Zero would read as total failure.
    assert citation_recall([], BLOCKS, entails_if_substring) is None


def test_precision_counts_a_citation_that_was_not_needed():
    """Block 3 contributes nothing: drop it and the sentence is still supported, so that
    citation is imprecise even though the sentence is fully supported."""
    text = "Proof of loss is due in 60 days [1][3]."

    assert citation_recall(claims(text, 3), BLOCKS, entails_if_substring) == 1.0
    assert citation_precision(claims(text, 3), BLOCKS, entails_if_substring) == 0.5


def test_a_sole_citation_that_supports_its_sentence_is_precise():
    text = "The limit is $30,000 [2]."

    assert citation_precision(claims(text, 3), BLOCKS, entails_if_substring) == 1.0


def test_citations_on_an_unsupported_sentence_are_not_counted_as_precise():
    # Precision is only meaningful where there is support to attribute.
    text = "Something unsupported [1][2]."

    assert citation_precision(claims(text, 3), BLOCKS, entails_if_substring) is None


# ---------------------------------------------------------- counts behind the rates



class TestCountsBehindTheRates:
    """A rate on its own cannot carry an interval. 5/16 and 50/160 are the same number
    and not the same evidence, and this arm's denominators are small enough that the
    difference decides whether anything can be concluded."""

    def test_recall_counts_give_supported_over_cited(self):
        blocks = {1: "The waiting period is 30 days.", 2: "Unrelated text about pets."}
        items = [
            Claim(sentence="The waiting period is 30 days. [1]", citations=[1]),
            Claim(sentence="Pets are covered. [2]", citations=[2]),
        ]
        def entails(premise, hypothesis):
            return hypothesis.rstrip(".").lower() in premise.lower()


        assert citation_recall_counts(items, blocks, entails) == (1, 2)

    def test_recall_counts_are_zero_over_zero_when_nothing_is_cited(self):
        assert citation_recall_counts([], {}, lambda p, h: True) == (0, 0)

    def test_the_counts_agree_with_the_rate_they_explain(self):
        blocks = {1: "A is true.", 2: "B is true."}
        items = [
            Claim(sentence="A is true. [1]", citations=[1]),
            Claim(sentence="C is true. [2]", citations=[2]),
        ]
        def entails(premise, hypothesis):
            return hypothesis.rstrip(".").lower() in premise.lower()


        supported, total = citation_recall_counts(items, blocks, entails)
        assert supported / total == citation_recall(items, blocks, entails)

    def test_precision_counts_give_needed_over_offered(self):
        blocks = {1: "The limit is $250,000.", 2: "Filler sentence."}
        items = [Claim(sentence="The limit is $250,000. [1][2]", citations=[1, 2])]
        def entails(premise, hypothesis):
            return "250,000" in premise and "250,000" in hypothesis


        # Both citations are offered; only block 1 is needed.
        assert citation_precision_counts(items, blocks, entails) == (1, 2)

    def test_precision_counts_agree_with_the_rate(self):
        blocks = {1: "The limit is $250,000.", 2: "Filler sentence."}
        items = [Claim(sentence="The limit is $250,000. [1][2]", citations=[1, 2])]
        def entails(premise, hypothesis):
            return "250,000" in premise and "250,000" in hypothesis


        needed, total = citation_precision_counts(items, blocks, entails)
        assert needed / total == citation_precision(items, blocks, entails)


# ------------------------------------------- hypotheses the judge can actually read


class TestHypothesisIsCleanedOfMarkerDebris:
    """The generator sometimes cites in subject position - "[1], [2], and provide
    relevant information." - so removing the markers leaves punctuation and a dangling
    conjunction rather than a statement. Nine of sixteen scored pairs looked like this,
    and one was the empty string. An entailment model cannot support a fragment, so those
    pairs failed by construction and the published rate was partly measuring the
    generator's citation placement."""

    def test_a_paragraph_break_inside_the_sentence_becomes_a_space(self):
        got = hypothesis_of("[2], [5]\n\nThe maximum payable is $30,000.")
        assert "\n" not in got
        assert got == "The maximum payable is $30,000."

    def test_leading_punctuation_and_connective_left_by_markers_are_stripped(self):
        assert hypothesis_of("[1],, and provide relevant information.") == (
            "provide relevant information."
        )

    def test_a_sentence_that_was_only_citations_becomes_empty(self):
        assert hypothesis_of("[1] [2] [3]") == ""

    def test_internal_runs_of_whitespace_collapse(self):
        assert hypothesis_of("The   limit  is\n$250,000.") == "The limit is $250,000."

    def test_ordinary_sentences_are_untouched_apart_from_the_markers(self):
        assert hypothesis_of("Proof of loss is due in 60 days. [1]") == (
            "Proof of loss is due in 60 days."
        )


class TestIsClaim:
    """Whether what survived marker removal can be judged at all.

    An earlier version of this rejected any remainder starting with "and", which threw
    away a real claim: "and According to the context, ... the revised due date will be 30
    days after the date on which the bill is mailed." That is a statement with a
    connective stuck to the front, and discarding it was a hand-made grammar rule getting
    the grammar wrong. The connective is now stripped as debris, and the only thing
    excluded is a hypothesis with nothing left in it - which no person could label and no
    passage could entail. Whether "provide relevant information" is supported is a
    judgment for the annotator, not for a regex."""

    def test_an_empty_hypothesis_is_not_a_claim(self):
        assert is_claim("") is False

    def test_whitespace_only_is_not_a_claim(self):
        assert is_claim("   ") is False

    def test_a_statement_behind_a_leading_connective_survives(self):
        text = hypothesis_of(
            "[1] and According to the context, the revised due date will be 30 days."
        )
        assert text == "According to the context, the revised due date will be 30 days."
        assert is_claim(text) is True

    def test_a_citation_referential_fragment_is_still_put_to_the_annotator(self):
        """It is not a claim about flood policy - it is a claim about the blocks - but a
        person can read it and decide. A regex deciding for them is what went wrong."""
        text = hypothesis_of("[1], [2] and provide relevant information.")
        assert text == "provide relevant information."
        assert is_claim(text) is True

    def test_an_ordinary_statement_is_a_claim(self):
        assert is_claim("The maximum payable is $30,000.") is True

    def test_a_statement_merely_containing_and_is_untouched(self):
        text = hypothesis_of("Coverage A and Coverage D are separate limits.")
        assert text == "Coverage A and Coverage D are separate limits."
