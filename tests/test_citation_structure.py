"""Whether the generator used its citations correctly, as a property of the string.

`docs/LIMITATIONS.md` records that the model is prompted to cite context numbers but
that whether it does so is not measured, and that no answer-quality metric is published
because the only local model grading its own output would be circular.

Nothing here grades an answer. These are facts about the text: did it cite, did it cite
a context that was actually supplied, how much of the context did it lean on. A parser
cannot be circular, so this half of the gap closes without a judge.
"""

from __future__ import annotations

from eval.citation_structure import (
    CitationUse,
    parse_citations,
    summarise,
    use_of_citations,
)


def test_a_single_marker_is_found():
    assert parse_citations("The limit is $250,000 [1].") == [1]


def test_several_markers_are_found_in_order_of_first_appearance():
    assert parse_citations("Both [2] and [1] apply, and [2] again.") == [2, 1]


def test_a_comma_separated_group_is_several_citations():
    """Models write [1, 2] as readily as [1][2]; counting it as one would undercount."""
    assert parse_citations("Coverage applies [1, 3].") == [1, 3]


def test_adjacent_markers_are_separate_citations():
    assert parse_citations("See [1][2][3].") == [1, 2, 3]


def test_prose_in_brackets_is_not_a_citation():
    assert parse_citations("See [see section 3] and [Appendix A].") == []


def test_an_answer_with_no_markers_cites_nothing():
    assert parse_citations("I don't know.") == []


def test_a_citation_outside_the_supplied_range_is_separated_out():
    # The context block is numbered 1..k. Anything else was invented.
    used = use_of_citations("Per [1] and [9].", n_contexts=5)

    assert used.in_range == [1]
    assert used.out_of_range == [9]
    assert used.cited_any is True


def test_zero_is_out_of_range_because_the_blocks_start_at_one():
    assert use_of_citations("Per [0].", n_contexts=5).out_of_range == [0]


def test_an_uncited_answer_is_recorded_as_such():
    used = use_of_citations("Flood damage is excluded.", n_contexts=5)

    assert used.cited_any is False
    assert used.in_range == []


def test_coverage_is_the_share_of_supplied_contexts_actually_used():
    used = use_of_citations("Per [1] and [2].", n_contexts=5)

    assert used.context_coverage == 0.4


def test_coverage_is_zero_when_no_contexts_were_supplied():
    # Guards a division by zero rather than reporting a fabricated rate.
    assert use_of_citations("Per [1].", n_contexts=0).context_coverage == 0.0


def test_a_repeated_citation_counts_once_towards_coverage():
    used = use_of_citations("Per [1], and again [1].", n_contexts=4)

    assert used.in_range == [1]
    assert used.context_coverage == 0.25


def test_the_summary_counts_answers_not_markers():
    rows = [
        use_of_citations("Per [1].", n_contexts=5),
        use_of_citations("No idea.", n_contexts=5),
        use_of_citations("Per [9].", n_contexts=5),
    ]

    summary = summarise(rows)

    assert summary["answers"] == 3
    assert summary["cited_any"] == 2
    assert summary["with_out_of_range"] == 1
    assert summary["uncited"] == 1


def test_the_summary_of_nothing_is_empty_rather_than_a_divide_by_zero():
    summary = summarise([])

    assert summary["answers"] == 0
    assert summary["uncited_rate"] == 0.0


def test_a_use_record_serialises_for_the_artifact():
    payload = use_of_citations("Per [1] and [9].", n_contexts=5).as_dict()

    assert payload["in_range"] == [1]
    assert payload["out_of_range"] == [9]
    assert payload["n_contexts"] == 5
    assert isinstance(
        CitationUse(
            **{
                k: payload[k]
                for k in ("cited", "in_range", "out_of_range", "n_contexts")
            }
        ),
        CitationUse,
    )


def test_the_summary_publishes_percentages_as_well_as_rates():
    """The docs quote percentages, and verify_artifacts matches documented text against
    the artefact. Storing only fractions would mean the published 27.8% is checked
    against 0.278 and never actually verified."""
    rows = [use_of_citations("Per [9].", n_contexts=5)] + [
        use_of_citations("Per [1].", n_contexts=5) for _ in range(3)
    ]

    summary = summarise(rows)

    assert summary["out_of_range_rate"] == 0.25
    assert summary["out_of_range_pct"] == 25.0
    # The out-of-range row covers nothing, so the mean is (0 + 0.2 + 0.2 + 0.2) / 4.
    assert summary["mean_context_coverage_pct"] == 15.0


def test_percentages_are_zero_on_an_empty_summary():
    summary = summarise([])

    assert summary["out_of_range_pct"] == 0.0
    assert summary["mean_context_coverage_pct"] == 0.0


# ------------------------------------------------- the prompt variant under test

from eval.prompt_variants import VARIANTS, ranged_prompt

BLOCKS = [
    {
        "text": "Proof of loss is due within 60 days.",
        "source": "Dwelling",
        "cfr_citation": "A",
    },
    {
        "text": "Increased Cost of Compliance is $30,000.",
        "source": "Dwelling",
        "cfr_citation": "B",
    },
]


def test_the_ranged_prompt_names_the_highest_valid_number():
    """The served prompt says "cite the numbers you used" and never says which numbers
    exist. The model answered with [6], [7] and [16] against five blocks."""
    assert "1 to 2" in ranged_prompt("How long do I have?", BLOCKS)


def test_the_ranged_prompt_still_carries_every_block_and_the_question():
    prompt = ranged_prompt("How long do I have?", BLOCKS)

    assert "Proof of loss is due within 60 days." in prompt
    assert "Increased Cost of Compliance is $30,000." in prompt
    assert "How long do I have?" in prompt
    assert "[1]" in prompt and "[2]" in prompt


def test_the_variant_table_includes_the_served_prompt_unchanged():
    """The comparison is only meaningful if one arm is what the service actually sends."""
    from src.rag import build_prompt

    assert VARIANTS["served"] is build_prompt
    assert set(VARIANTS) == {"served", "ranged"}
