"""
Ground-truth labels and their validator.

The committed labels must validate. Equally important, the validator must *fail* on each
kind of breakage - a validator that silently passes bad labels is worse than none, because
it makes the benchmark look checked.
"""
import json

import pytest

from eval.ground_truth import (
    CATEGORIES,
    Question,
    load_questions,
    question_set_hash,
    summarise,
    validate,
)
from scripts.build_ground_truth import assign_split
from src.corpus import chunk_corpus, load_corpus

SIZE, OVERLAP = 800, 120


@pytest.fixture(scope="module")
def documents():
    return load_corpus()


@pytest.fixture(scope="module")
def chunks(documents):
    return chunk_corpus(documents, SIZE, OVERLAP)


@pytest.fixture(scope="module")
def questions():
    return load_questions()


def _raw(question: Question) -> dict:
    return {
        "question_id": question.question_id,
        "question": question.question,
        "answerable": question.answerable,
        "category": question.category,
        "subdomain": question.subdomain,
        "difficulty": question.difficulty,
        "split": question.split,
        "relevant_document_ids": list(question.relevant_document_ids),
        "relevant_chunk_ids": list(question.relevant_chunk_ids),
        "evidence_spans": [
            {"document_id": s.document_id, "start": s.start, "end": s.end}
            for s in question.evidence_spans
        ],
        "gold_answer": question.gold_answer,
        "distractor_document_ids": list(question.distractor_document_ids),
        "provenance": question.provenance,
    }


class TestCommittedLabels:
    def test_labels_validate_against_the_corpus(self, questions, documents, chunks):
        assert validate(questions, documents, chunks) == []

    def test_expected_size_and_split(self, questions):
        stats = summarise(questions)
        assert stats["total"] == 40
        assert stats["answerable"] == 32
        assert stats["unanswerable"] == 8
        assert 0.15 <= stats["unanswerable_share"] <= 0.30

    def test_every_category_is_represented(self, questions):
        used = {q.category for q in questions}
        assert used <= CATEGORIES
        # The eight designed question kinds must all be present.
        assert used == CATEGORIES

    def test_both_splits_are_populated(self, questions):
        stats = summarise(questions)
        assert stats["by_split"]["dev"] > 0
        assert stats["by_split"]["test"] > 0

    def test_all_three_forms_are_targeted(self, questions):
        targeted = {d for q in questions for d in q.relevant_document_ids}
        assert targeted == {
            "nfip-sfip-dwelling",
            "nfip-sfip-general-property",
            "nfip-sfip-rcbap",
        }

    def test_evidence_spans_land_on_real_text(self, questions, documents):
        by_id = {d.document_id: d for d in documents}
        for question in questions:
            for span in question.evidence_spans:
                assert span.text_from(by_id[span.document_id]).strip()

    def test_gold_answers_do_not_name_the_form(self, questions):
        """A gold answer that names the form would leak the correct document."""
        leaks = ["Dwelling Form", "General Property Form", "RCBAP"]
        for question in questions:
            if question.gold_answer:
                for leak in leaks:
                    assert leak not in question.gold_answer, question.question_id

    def test_unanswerable_questions_have_no_evidence(self, questions):
        for question in questions:
            if not question.answerable:
                assert not question.relevant_chunk_ids
                assert not question.relevant_document_ids
                assert not question.evidence_spans
                assert question.gold_answer is None

    def test_question_set_hash_is_stable(self, questions):
        assert question_set_hash(questions) == question_set_hash(load_questions())

    def test_question_set_hash_changes_with_labels(self, questions):
        mutated = list(questions)
        first = mutated[0]
        mutated[0] = Question.from_dict({**_raw(first), "relevant_chunk_ids": ["different"]})
        assert question_set_hash(mutated) != question_set_hash(questions)


class TestDeterministicSplit:
    def test_split_is_reproducible(self, questions):
        for question in questions:
            assert assign_split(question.question_id, question.category) == question.split

    def test_split_only_returns_known_values(self):
        assert {assign_split(f"q-{i}", "single_chunk") for i in range(50)} <= {"dev", "test"}

    def test_split_is_not_all_one_side(self):
        assigned = [assign_split(f"q-{i}", "single_chunk") for i in range(60)]
        assert 0 < assigned.count("dev") < 60


class TestValidatorCatchesBreakage:
    """Each test breaks one thing and asserts the validator notices."""

    def test_missing_chunk_reference(self, questions, documents, chunks):
        broken = Question.from_dict({
            **_raw(next(q for q in questions if q.answerable)),
            "relevant_chunk_ids": ["nfip-sfip-dwelling#deadbeef0000"],
        })
        problems = validate([broken], documents, chunks)
        assert any("relevant chunk does not exist" in p for p in problems)

    def test_unknown_document(self, questions, documents, chunks):
        source = next(q for q in questions if q.answerable)
        broken = Question.from_dict({**_raw(source), "relevant_document_ids": ["not-a-doc"]})
        problems = validate([broken], documents, chunks)
        assert any("unknown document" in p for p in problems)

    def test_evidence_span_outside_the_document(self, questions, documents, chunks):
        source = next(q for q in questions if q.evidence_spans)
        raw = _raw(source)
        raw["evidence_spans"] = [
            {"document_id": source.evidence_spans[0].document_id, "start": 10, "end": 10_000_000}
        ]
        problems = validate([Question.from_dict(raw)], documents, chunks)
        assert any("outside" in p for p in problems)

    def test_inverted_evidence_span(self, questions, documents, chunks):
        source = next(q for q in questions if q.evidence_spans)
        raw = _raw(source)
        raw["evidence_spans"] = [
            {"document_id": source.evidence_spans[0].document_id, "start": 500, "end": 100}
        ]
        problems = validate([Question.from_dict(raw)], documents, chunks)
        assert any("outside" in p for p in problems)

    def test_answerable_with_no_evidence(self, questions, documents, chunks):
        source = next(q for q in questions if q.answerable)
        raw = _raw(source)
        raw["relevant_chunk_ids"] = []
        problems = validate([Question.from_dict(raw)], documents, chunks)
        assert any("answerable but has no relevant chunks" in p for p in problems)

    def test_unanswerable_carrying_relevant_chunks(self, questions, documents, chunks):
        source = next(q for q in questions if not q.answerable)
        donor = next(q for q in questions if q.answerable)
        raw = _raw(source)
        raw["relevant_chunk_ids"] = list(donor.relevant_chunk_ids)
        problems = validate([Question.from_dict(raw)], documents, chunks)
        assert any("unanswerable but carries relevant chunks" in p for p in problems)

    def test_duplicate_question_ids(self, questions, documents, chunks):
        source = next(q for q in questions if q.answerable)
        problems = validate([source, source], documents, chunks)
        assert any("duplicate question_id" in p for p in problems)

    def test_duplicate_question_text(self, questions, documents, chunks):
        source = next(q for q in questions if q.answerable)
        twin = Question.from_dict({**_raw(source), "question_id": "nfip-999"})
        problems = validate([source, twin], documents, chunks)
        assert any("duplicate question text" in p for p in problems)

    def test_unknown_category(self, questions, documents, chunks):
        source = next(q for q in questions if q.answerable)
        broken = Question.from_dict({**_raw(source), "category": "invented"})
        assert any("unknown category" in p for p in validate([broken], documents, chunks))

    def test_unknown_split(self, questions, documents, chunks):
        source = next(q for q in questions if q.answerable)
        broken = Question.from_dict({**_raw(source), "split": "holdout"})
        assert any("unknown split" in p for p in validate([broken], documents, chunks))

    def test_document_both_relevant_and_distractor(self, questions, documents, chunks):
        source = next(q for q in questions if q.answerable)
        raw = _raw(source)
        raw["distractor_document_ids"] = list(source.relevant_document_ids)
        problems = validate([Question.from_dict(raw)], documents, chunks)
        assert any("both relevant and distractor" in p for p in problems)

    def test_chunk_from_an_unlisted_document(self, questions, documents, chunks):
        source = next(q for q in questions if q.answerable)
        other = next(
            c for c in chunks if c.document_id not in source.relevant_document_ids
        )
        raw = _raw(source)
        raw["relevant_chunk_ids"] = [other.chunk_id]
        problems = validate([Question.from_dict(raw)], documents, chunks)
        assert any("not listed in relevant_document_ids" in p for p in problems)

    def test_multi_chunk_with_one_chunk(self, questions, documents, chunks):
        source = next(q for q in questions if q.answerable)
        raw = _raw(source)
        raw["category"] = "multi_chunk"
        raw["relevant_chunk_ids"] = source.relevant_chunk_ids[:1]
        problems = validate([Question.from_dict(raw)], documents, chunks)
        assert any("multi_chunk" in p for p in problems)

    def test_missing_provenance(self, questions, documents, chunks):
        source = next(q for q in questions if q.answerable)
        broken = Question.from_dict({**_raw(source), "provenance": ""})
        assert any("missing provenance" in p for p in validate([broken], documents, chunks))


class TestGroundTruthFile:
    def test_is_valid_jsonl(self):
        path = "eval/ground_truth/nfip_questions.jsonl"
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == 40
        assert all("question_id" in row for row in rows)

    def test_no_benchmark_question_appears_in_the_finetuning_data(self, questions):
        """The synthetic fine-tuning set must never overlap this benchmark."""
        with open("archive/finetune/lora_finetune.ipynb", encoding="utf-8") as f:
            notebook = json.load(f)
        training_text = " ".join(
            "".join(cell["source"]) for cell in notebook["cells"]
        ).lower()
        for question in questions:
            assert question.question.lower() not in training_text
