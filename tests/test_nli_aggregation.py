"""Scoring a long premise the way sentence-pair NLI models expect to be used.

Both judges failed their control on 800-character legalistic premises: the base model
declined support that was there, the large one accepted support that was not. The
literature's answer to a premise longer than the training distribution is to break it
into sentences, score each against the hypothesis, and aggregate
(arXiv:2204.07447), and to choose a threshold per task rather than take an argmax.

Both are pure functions of a scorer, so both are tested against a stated fake.
"""

from __future__ import annotations

from eval.nli_aggregation import (
    batch_plan,
    best_scores,
    best_sentence_score,
    chunk_covering,
    paraphrase_control,
    youden_threshold,
)


def scorer(table: dict[tuple[str, str], float]):
    return lambda premise, hypothesis: table.get((premise, hypothesis), 0.0)


def test_the_premise_is_scored_one_sentence_at_a_time():
    """A sentence-pair model sees sentence pairs, which is what it was trained on."""
    premise = "Irrelevant opening. Proof of loss is due in 60 days. Another clause."
    score = scorer({("Proof of loss is due in 60 days.", "Due in 60 days."): 0.9})

    assert best_sentence_score(premise, "Due in 60 days.", score) == 0.9


def test_the_best_sentence_wins_not_the_average():
    """Support from one sentence is support. Averaging would let a long block of
    unrelated text bury the clause that actually says it."""
    premise = "A. B. C."
    score = scorer({("A.", "h"): 0.1, ("B.", "h"): 0.8, ("C.", "h"): 0.2})

    assert best_sentence_score(premise, "h", score) == 0.8


def test_a_premise_with_no_sentences_scores_zero():
    assert best_sentence_score("", "h", scorer({})) == 0.0


def test_the_whole_premise_is_also_offered_to_the_scorer():
    """Some claims are supported only by two clauses together, which no single sentence
    entails. The undivided premise stays in the pool so those are not lost."""
    premise = "First half. Second half."
    score = scorer({("First half. Second half.", "h"): 0.7})

    assert best_sentence_score(premise, "h", score) == 0.7


def test_youden_picks_the_cut_that_best_separates_the_two_sets():
    positives = [0.9, 0.8, 0.7]
    negatives = [0.2, 0.1, 0.3]

    cut = youden_threshold(positives, negatives)

    assert all(p >= cut for p in positives)
    assert all(n < cut for n in negatives)


def test_youden_returns_none_when_either_side_is_empty():
    assert youden_threshold([], [0.1]) is None
    assert youden_threshold([0.9], []) is None


def test_youden_prefers_separation_over_catching_everything():
    """A cut at zero catches every positive and every negative too. J is sensitivity
    plus specificity minus one, so it refuses that trade."""
    positives = [0.9, 0.85]
    negatives = [0.1, 0.8]

    cut = youden_threshold(positives, negatives)

    assert cut > 0.8


# ------------------------------------------- a control shaped like the task it calibrates


ROWS = [
    {
        "question_id": "q1",
        "answerable": True,
        "gold_answer": "A proof of loss must be sent within 60 days.",
        "relevant_chunk_ids": ["d#a"],
    },
    {
        "question_id": "q2",
        "answerable": True,
        "gold_answer": "The limit is $30,000.",
        "relevant_chunk_ids": ["d#b"],
    },
    {"question_id": "q3", "answerable": False, "gold_answer": "", "relevant_chunk_ids": []},
]
TEXTS = {"d#a": "Send proof of loss within 60 days.", "d#b": "ICC is capped at $30,000.", "d#c": "Unrelated clause."}


def test_a_positive_pairs_a_gold_answer_with_a_chunk_that_supports_it():
    positives, _ = paraphrase_control(ROWS, TEXTS)

    assert ("Send proof of loss within 60 days.", "A proof of loss must be sent within 60 days.") in positives


def test_a_negative_pairs_the_same_answer_with_a_chunk_labelled_irrelevant():
    _, negatives = paraphrase_control(ROWS, TEXTS)

    assert all(hyp in {r["gold_answer"] for r in ROWS} for _, hyp in negatives)
    assert all(prem != TEXTS["d#a"] or hyp != ROWS[0]["gold_answer"] for prem, hyp in negatives)


def test_unanswerable_questions_are_skipped():
    """They have no gold answer, so there is nothing a chunk could support."""
    positives, negatives = paraphrase_control(ROWS, TEXTS)

    assert all(hyp for _, hyp in positives + negatives)


def test_the_control_is_paraphrase_not_copy():
    """The point of this control: no positive is the hypothesis literally inside its
    premise, because that is the easy case and calibrating on it sets the cut far too
    high for the paraphrased claims a citation metric actually scores."""
    positives, _ = paraphrase_control(ROWS, TEXTS)

    assert all(hyp not in prem for prem, hyp in positives)


# ----------------------------------- positives must actually contain the labelled evidence


class Chunk:
    def __init__(self, cid, doc, start, end, text):
        self.chunk_id, self.document_id, self.start, self.end, self.text = cid, doc, start, end, text


CHUNKS = [
    Chunk("d#1", "d", 0, 100, "early text"),
    Chunk("d#2", "d", 100, 200, "the clause that supports the claim"),
    Chunk("e#1", "e", 0, 100, "other document"),
]


def test_the_chunk_holding_the_evidence_span_is_chosen():
    """relevant_chunk_ids lists several chunks and only some carry the evidence. Taking
    the first pairs a claim with a passage that does not support it, which depresses
    sensitivity and mis-calibrates the cut all over again."""
    span = {"document_id": "d", "start": 120, "end": 140}

    assert chunk_covering(span, CHUNKS).chunk_id == "d#2"


def test_a_span_in_another_document_is_not_matched():
    assert chunk_covering({"document_id": "e", "start": 150, "end": 160}, CHUNKS) is None


def test_a_span_nothing_covers_returns_nothing():
    assert chunk_covering({"document_id": "d", "start": 900, "end": 950}, CHUNKS) is None


# --------------------------------------------- scoring many pairs in one pass


def batch_scorer(table):
    """Records how many batches it was asked for, so the test can see them collapse."""
    calls = []

    def score_many(pairs):
        calls.append(len(pairs))
        return [table.get(pair, 0.0) for pair in pairs]

    return score_many, calls


def test_every_premise_part_of_every_pair_is_scored_in_one_call():
    """One forward pass per batch, not per sentence. At ~1300 single-item passes the
    run does not finish on the hardware this is meant to run on."""
    score_many, calls = batch_scorer({("B.", "h1"): 0.8, ("D.", "h2"): 0.6})

    out = best_scores([("A. B.", "h1"), ("C. D.", "h2")], score_many)

    assert out == [0.8, 0.6]
    assert len(calls) == 1


def test_the_best_part_still_wins_per_pair():
    score_many, _ = batch_scorer({("A.", "h"): 0.2, ("B.", "h"): 0.9})

    assert best_scores([("A. B.", "h")], score_many) == [0.9]


def test_an_empty_premise_scores_zero_without_reaching_the_model():
    score_many, calls = batch_scorer({})

    assert best_scores([("", "h")], score_many) == [0.0]
    assert calls == [] or calls == [0]


# ------------------------------------------------------------------ batch planning


def test_short_pairs_batch_together_and_long_ones_do_not():
    """Cost is batch x padded length, and padding is the longest member. A 2 GiB card
    OOMs on a fixed count because one 512-token premise drags 31 short sentences up
    with it."""
    plan = batch_plan([10, 10, 10, 10, 500], budget=100)

    assert [4, 4, 4, 4] in [sorted(b) for b in plan] or [0, 1, 2, 3] in plan
    long_batch = next(b for b in plan if 4 in b)
    assert long_batch == [4], "a 500-token pair must go alone under a 100-token budget"


def test_every_index_appears_exactly_once():
    plan = batch_plan([5, 300, 7, 120, 9], budget=64)

    assert sorted(i for b in plan for i in b) == [0, 1, 2, 3, 4]


def test_similar_lengths_are_grouped_so_padding_is_small():
    """Sorting by length is what makes the budget tight rather than nominal."""
    plan = batch_plan([500, 10, 500, 10], budget=1000)

    assert [1, 3] in plan, f"the two short pairs should share a batch: {plan}"


def test_a_pair_over_budget_still_gets_its_own_batch():
    assert batch_plan([5000], budget=100) == [[0]]


def test_no_pairs_means_no_batches():
    assert batch_plan([], budget=100) == []
