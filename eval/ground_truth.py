"""
Ground-truth schema for retrieval evaluation, and its validator.

A label is only worth as much as its weakest reference. If a label points at a chunk that
no longer exists, or an evidence span that falls outside the document, the metric computed
from it is silently wrong rather than loudly broken. Everything here exists to make that
impossible: every reference is resolved against the real corpus, and anything that does not
resolve fails the run.

Validate with:
    python -m eval.validate_ground_truth
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from src.corpus import Chunk, Document
from src.paths import GROUND_TRUTH_PATH

#: Controlled vocabularies. Closed on purpose - a typo in a category silently creates a new
#: bucket and quietly distorts any per-category breakdown.
CATEGORIES = {
    "single_chunk",        # one passage answers it outright
    "multi_chunk",         # needs two or more passages combined
    "near_miss",           # all three forms look relevant; only one is correct
    "exclusion",           # what is NOT covered
    "numeric_limit",       # a monetary cap, deductible, or percentage
    "time_period",         # a deadline or duration
    "form_difference",     # a provision that differs between forms
    "unanswerable",        # not answerable from this corpus
}

DIFFICULTIES = {"easy", "medium", "hard"}

SUBDOMAINS = {"flood/us"}

#: Documents the benchmark may reference.
KNOWN_DOCUMENTS = {
    "nfip-sfip-dwelling",
    "nfip-sfip-general-property",
    "nfip-sfip-rcbap",
}

SPLITS = {"dev", "test"}


@dataclass(frozen=True)
class EvidenceSpan:
    document_id: str
    start: int
    end: int

    def text_from(self, document: Document) -> str:
        return document.text[self.start:self.end]


@dataclass(frozen=True)
class Question:
    question_id: str
    question: str
    answerable: bool
    category: str
    subdomain: str
    difficulty: str
    split: str
    relevant_document_ids: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = field(default_factory=list)
    gold_answer: str | None = None
    distractor_document_ids: list[str] = field(default_factory=list)
    provenance: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> Question:
        return cls(
            question_id=raw["question_id"],
            question=raw["question"],
            answerable=raw["answerable"],
            category=raw["category"],
            subdomain=raw["subdomain"],
            difficulty=raw["difficulty"],
            split=raw["split"],
            relevant_document_ids=raw.get("relevant_document_ids", []),
            relevant_chunk_ids=raw.get("relevant_chunk_ids", []),
            evidence_spans=[EvidenceSpan(**s) for s in raw.get("evidence_spans", [])],
            gold_answer=raw.get("gold_answer"),
            distractor_document_ids=raw.get("distractor_document_ids", []),
            provenance=raw.get("provenance", ""),
        )


def load_questions(path: Path = GROUND_TRUTH_PATH) -> list[Question]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            questions.append(Question.from_dict(json.loads(line)))
    return questions


def question_set_hash(questions: list[Question]) -> str:
    """Hash of the question IDs and their targets, for stamping a reference run."""
    parts = [
        f"{q.question_id}|{q.answerable}|{','.join(sorted(q.relevant_chunk_ids))}"
        for q in sorted(questions, key=lambda q: q.question_id)
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def validate(
    questions: list[Question],
    documents: list[Document],
    chunks: list[Chunk],
) -> list[str]:
    """Return a list of problems. An empty list means the labels are internally sound.

    Each rule below corresponds to a way a benchmark can look fine and measure nothing.
    """
    problems: list[str] = []

    documents_by_id = {d.document_id: d for d in documents}
    chunk_ids = {c.chunk_id for c in chunks}
    chunks_by_id = {c.chunk_id: c for c in chunks}

    seen_ids: set[str] = set()
    seen_questions: dict[str, str] = {}

    for q in questions:
        where = f"[{q.question_id}]"

        # --- identity -------------------------------------------------------------
        if q.question_id in seen_ids:
            problems.append(f"{where} duplicate question_id")
        seen_ids.add(q.question_id)

        normalised = q.question.strip().lower()
        if normalised in seen_questions:
            problems.append(
                f"{where} duplicate question text, already used by "
                f"{seen_questions[normalised]}"
            )
        else:
            seen_questions[normalised] = q.question_id

        if not q.question.strip():
            problems.append(f"{where} empty question")

        # --- controlled vocabularies ----------------------------------------------
        if q.category not in CATEGORIES:
            problems.append(f"{where} unknown category {q.category!r}")
        if q.subdomain not in SUBDOMAINS:
            problems.append(f"{where} unknown subdomain {q.subdomain!r}")
        if q.difficulty not in DIFFICULTIES:
            problems.append(f"{where} unknown difficulty {q.difficulty!r}")
        if q.split not in SPLITS:
            problems.append(f"{where} unknown split {q.split!r}")

        # --- references must resolve ----------------------------------------------
        for document_id in q.relevant_document_ids + q.distractor_document_ids:
            if document_id not in KNOWN_DOCUMENTS:
                problems.append(f"{where} unknown document {document_id!r}")
            elif document_id not in documents_by_id:
                problems.append(f"{where} document {document_id!r} not present in corpus")

        for chunk_id in q.relevant_chunk_ids:
            if chunk_id not in chunk_ids:
                problems.append(f"{where} relevant chunk does not exist: {chunk_id}")
            else:
                owner = chunks_by_id[chunk_id].document_id
                if owner not in q.relevant_document_ids:
                    problems.append(
                        f"{where} chunk {chunk_id} belongs to {owner}, which is not "
                        "listed in relevant_document_ids"
                    )

        for span in q.evidence_spans:
            document = documents_by_id.get(span.document_id)
            if document is None:
                problems.append(f"{where} evidence span in unknown document {span.document_id!r}")
                continue
            if span.start < 0 or span.end > len(document.text) or span.start >= span.end:
                problems.append(
                    f"{where} evidence span [{span.start}:{span.end}] is outside "
                    f"{span.document_id} (length {len(document.text)})"
                )

        # --- answerable / unanswerable invariants ---------------------------------
        if q.answerable:
            if not q.relevant_chunk_ids:
                problems.append(f"{where} answerable but has no relevant chunks")
            if not q.relevant_document_ids:
                problems.append(f"{where} answerable but has no relevant documents")
            if not q.gold_answer:
                problems.append(f"{where} answerable but has no gold answer")
            if q.category == "unanswerable":
                problems.append(f"{where} category 'unanswerable' but answerable is true")
        else:
            if q.relevant_chunk_ids:
                problems.append(
                    f"{where} unanswerable but carries relevant chunks "
                    f"({len(q.relevant_chunk_ids)})"
                )
            if q.relevant_document_ids:
                problems.append(f"{where} unanswerable but carries relevant documents")
            if q.evidence_spans:
                problems.append(f"{where} unanswerable but carries evidence spans")
            if q.category != "unanswerable":
                problems.append(
                    f"{where} answerable is false but category is {q.category!r}"
                )

        # --- consistency ----------------------------------------------------------
        overlap = set(q.relevant_document_ids) & set(q.distractor_document_ids)
        if overlap:
            problems.append(
                f"{where} document is both relevant and distractor: {sorted(overlap)}"
            )

        if q.category == "multi_chunk" and len(q.relevant_chunk_ids) < 2:
            problems.append(
                f"{where} category 'multi_chunk' but only "
                f"{len(q.relevant_chunk_ids)} relevant chunk(s)"
            )

        if not q.provenance:
            problems.append(f"{where} missing provenance")

    return problems


def summarise(questions: list[Question]) -> dict:
    """Counts used by the reference run and the report."""
    answerable = [q for q in questions if q.answerable]
    unanswerable = [q for q in questions if not q.answerable]

    by_category: dict[str, int] = {}
    by_split: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    for q in questions:
        by_category[q.category] = by_category.get(q.category, 0) + 1
        by_split[q.split] = by_split.get(q.split, 0) + 1
        by_difficulty[q.difficulty] = by_difficulty.get(q.difficulty, 0) + 1

    return {
        "total": len(questions),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "unanswerable_share": round(len(unanswerable) / len(questions), 4) if questions else 0.0,
        "by_category": dict(sorted(by_category.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_difficulty": dict(sorted(by_difficulty.items())),
    }
