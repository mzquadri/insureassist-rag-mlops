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
    citation_precision,
    citation_recall,
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
