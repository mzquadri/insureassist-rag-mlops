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
