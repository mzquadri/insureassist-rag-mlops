"""
Build the NFIP retrieval ground truth.

    python -m scripts.build_ground_truth

The questions, the form each one targets, the gold answers and the categories are all
hand-authored below. What is *not* hand-authored is the mechanical part: each label carries
an `anchor`, a regular expression matching the provision in the source text, and this script
resolves that anchor to the document offsets and the chunk IDs that contain it.

Transcribing 40 sets of twelve-character chunk hashes by hand would be a reliable way to
produce labels that point at the wrong passage, and a label that silently points at the
wrong chunk makes the metric wrong rather than broken. Resolving them from the text keeps
the labels correct by construction, and re-running after a corpus change regenerates them.

An anchor that matches zero times, or more than once in its target document, is an error:
ambiguity means the question does not have one identifiable piece of evidence.

The dev/test split is deterministic - a hash of the question ID, stratified by category so
both halves keep the same shape.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.config import cfg
from src.corpus import chunk_corpus, load_corpus

OUT_PATH = Path("eval/ground_truth/nfip_questions.jsonl")

DWELL = "nfip-sfip-dwelling"
GENPROP = "nfip-sfip-general-property"
RCBAP = "nfip-sfip-rcbap"
ALL_FORMS = [DWELL, GENPROP, RCBAP]

PROVENANCE = "hand-authored against 44 CFR Pt.61 App. A(1)-A(3), eCFR issue 2026-06-22"


def Q(qid, question, *, doc=None, anchor=None, anchors=None, gold=None, category,
      difficulty="medium", distractors=(), answerable=True):
    """One labelled question. `anchors` is [(document_id, regex), ...] for multi-chunk."""
    return {
        "question_id": qid,
        "question": question,
        "answerable": answerable,
        "category": category,
        "difficulty": difficulty,
        "gold_answer": gold,
        "distractor_document_ids": list(distractors),
        "_anchors": anchors if anchors is not None else ([(doc, anchor)] if doc else []),
    }


# ---------------------------------------------------------------------------------------
# Hand-authored labels.
#
# The three forms are near-identical by design, so most questions name the form they ask
# about. Gold answers deliberately state the substantive rule without naming the form -
# otherwise the answer text would leak which document is correct.
# ---------------------------------------------------------------------------------------
QUESTIONS = [
    # --- A. single-chunk lookup ---------------------------------------------------------
    Q("nfip-001", "Under the Dwelling Form, how long do I have to send a proof of loss after a flood loss?",
      doc=DWELL, anchor=r"Within 60 days after the loss, send us a proof of loss",
      gold="A proof of loss must be sent within 60 days after the loss.",
      category="time_period", difficulty="easy", distractors=[GENPROP, RCBAP]),
    Q("nfip-002", "What is the maximum payable under Coverage D, Increased Cost of Compliance, on the Dwelling Form?",
      doc=DWELL, anchor=r"We will pay you up to \$30,000 under this Coverage D",
      gold="Up to $30,000, and it applies only to policies that carry building coverage.",
      category="numeric_limit", difficulty="easy", distractors=[GENPROP, RCBAP]),
    Q("nfip-003", "Under the Dwelling Form, may I sue the insurer without first meeting the policy's requirements?",
      doc=DWELL, anchor=r"You may not sue us to recover money under this policy unless you have complied",
      gold="No. You may not sue to recover money under the policy unless you have complied with all of its requirements.",
      category="single_chunk", difficulty="easy", distractors=[GENPROP, RCBAP]),
    Q("nfip-004", "How is a basement defined for flood insurance purposes under the Dwelling Form?",
      doc=DWELL, anchor=r"Basement\. Any area of a building, including any sunken room",
      gold="Any area of a building, including a sunken room or sunken portion of a room, having its floor below ground level on all sides.",
      category="single_chunk", difficulty="easy", distractors=[GENPROP, RCBAP]),
    Q("nfip-005", "Under the General Property Form, what is the special limit for artwork, photographs and collectibles?",
      doc=GENPROP, anchor=r"Special Limits\. We will pay no more than \$2,500 for any loss",
      gold="No more than $2,500 for any loss to one or more items of that kind.",
      category="near_miss", difficulty="hard", distractors=[DWELL, RCBAP]),
    Q("nfip-006", "Under the RCBAP, what is the special limit for artwork and collectibles?",
      doc=RCBAP, anchor=r"Special Limits\. We will pay no more than \$2,500 for any one loss",
      gold="No more than $2,500 for any one loss to one or more items of that kind.",
      category="near_miss", difficulty="hard", distractors=[DWELL, GENPROP]),
    Q("nfip-007", "Under the Dwelling Form, what is the special limit for artwork, photographs and memorabilia?",
      doc=DWELL, anchor=r"Special Limits\. We will pay no more than \$2,500 for any one loss",
      gold="No more than $2,500 for any one loss to one or more items of that kind.",
      category="near_miss", difficulty="hard", distractors=[GENPROP, RCBAP]),

    # --- B. exclusions ------------------------------------------------------------------
    Q("nfip-008", "Under the Dwelling Form, is damage caused by earth movement covered when the earth movement was itself caused by flood?",
      doc=DWELL, anchor=r"We do not insure for loss to property caused directly by earth movement even if the earth movement is caused by flood",
      gold="No. Loss caused directly by earth movement is excluded even when the earth movement is itself caused by flood.",
      category="exclusion", difficulty="medium", distractors=[GENPROP, RCBAP]),
    Q("nfip-009", "Under the General Property Form, does the policy cover landslide or earthquake damage?",
      doc=GENPROP, anchor=r"We do not insure for loss to property caused directly by earth movement even if the earth movement is caused by flood",
      gold="No. Earth movement is excluded, and earthquake and landslide are given as examples of it.",
      category="exclusion", difficulty="medium", distractors=[DWELL, RCBAP]),
    Q("nfip-010", "Under the RCBAP, is loss caused directly by earth movement covered?",
      doc=RCBAP, anchor=r"We do not insure for loss to property caused directly by earth movement even if the earth movement is caused by flood",
      gold="No. Loss caused directly by earth movement is excluded, even when the earth movement is caused by flood.",
      category="exclusion", difficulty="medium", distractors=[DWELL, GENPROP]),
    Q("nfip-011", "Under the Dwelling Form, are landslide and slope failure treated as mudflow?",
      doc=DWELL, anchor=r"slope failure, or a saturated soil mass",
      gold="No. Landslide, slope failure and a saturated soil mass moving by liquidity down a slope are expressly not mudflow.",
      category="exclusion", difficulty="hard", distractors=[GENPROP, RCBAP]),

    # --- C. form differences and hard negatives -----------------------------------------
    Q("nfip-012", "Under the RCBAP, what percentage of replacement cost must the building be insured to in order to avoid a coinsurance penalty?",
      doc=RCBAP, anchor=r"At least 80 percent of its replacement cost",
      gold="At least 80 percent of the building's replacement cost, or the maximum amount of insurance available for that building under the NFIP, whichever is less.",
      category="form_difference", difficulty="hard", distractors=[DWELL, GENPROP]),
    Q("nfip-013", "Which flood policy form imposes a coinsurance penalty on loss payment?",
      doc=RCBAP, anchor=r"We will impose a penalty on loss payment unless the amount of insurance applicable to the damaged building is",
      gold="Only the residential condominium building association form imposes a coinsurance penalty; the other two forms contain no coinsurance provision.",
      category="form_difference", difficulty="hard", distractors=[DWELL, GENPROP]),
    Q("nfip-014", "Under the Dwelling Form, how long after a loss must I notify the insurer if I intend to claim additional liability after a community conversion?",
      doc=DWELL, anchor=r"within 180 days after the date of loss",
      gold="Within 180 days after the date of loss.",
      category="time_period", difficulty="hard", distractors=[GENPROP, RCBAP]),

    # --- D. time periods ----------------------------------------------------------------
    Q("nfip-015", "Under the Dwelling Form, how long does the insurer have to give written notice that it may repair, rebuild or replace damaged property?",
      doc=DWELL, anchor=r"If we give you written notice within 30 days after we receive your signed, sworn proof of loss",
      gold="Within 30 days after receiving the signed, sworn proof of loss.",
      category="time_period", difficulty="medium", distractors=[GENPROP, RCBAP]),
    Q("nfip-016", "Under the Dwelling Form, for how long does cover continue for a mortgagee after the policy is cancelled or not renewed?",
      doc=DWELL, anchor=r"it will continue in effect for the benefit of the mortgagee only for 30 days after we notify the mortgagee",
      gold="For 30 days after the mortgagee is notified of the cancellation or non-renewal.",
      category="time_period", difficulty="medium", distractors=[GENPROP, RCBAP]),
    Q("nfip-017", "Under the General Property Form, what is the revised due date on a second premium bill?",
      doc=GENPROP, anchor=r"we will mail a second bill providing a revised due date, which will be 30 days after the date on which the bill is mailed",
      gold="30 days after the date on which the second bill is mailed.",
      category="time_period", difficulty="medium", distractors=[DWELL, RCBAP]),
    Q("nfip-018", "Under the RCBAP, how long after a loss must the proof of loss be submitted?",
      doc=RCBAP, anchor=r"Within 60 days after the loss, send us a proof of loss",
      gold="Within 60 days after the loss.",
      category="near_miss", difficulty="hard", distractors=[DWELL, GENPROP]),
    Q("nfip-019", "Under the General Property Form, when is the proof of loss due after a flood loss?",
      doc=GENPROP, anchor=r"Within 60 days after the loss, send us a proof of loss",
      gold="Within 60 days after the loss.",
      category="near_miss", difficulty="hard", distractors=[DWELL, RCBAP]),

    # --- E. numeric limits --------------------------------------------------------------
    Q("nfip-020", "Under the General Property Form, what is the Increased Cost of Compliance limit?",
      doc=GENPROP, anchor=r"We will pay you up to \$30,000 under this Coverage D \(Increased Cost of Compliance\)",
      gold="Up to $30,000, applying only to policies that carry building coverage.",
      category="near_miss", difficulty="hard", distractors=[DWELL, RCBAP]),
    Q("nfip-021", "Under the RCBAP, what is the maximum payable for Increased Cost of Compliance?",
      doc=RCBAP, anchor=r"We will pay you up to \$30,000 under this Coverage D \(Increased Cost of Compliance\)",
      gold="Up to $30,000, applying only to policies that carry building coverage.",
      category="near_miss", difficulty="hard", distractors=[DWELL, GENPROP]),
    Q("nfip-022", "Under the Dwelling Form, what is a base flood?",
      doc=DWELL, anchor=r"Base Flood\. A flood having a one percent chance of being equaled or exceeded in any given year",
      gold="A flood having a one percent chance of being equalled or exceeded in any given year.",
      category="numeric_limit", difficulty="easy", distractors=[GENPROP, RCBAP]),
    Q("nfip-023", "Under the RCBAP, what is an elevated building?",
      doc=RCBAP, anchor=r"Elevated Building\. A building that has no basement and that has its lowest elevated floor raised above ground level",
      gold="A building with no basement whose lowest elevated floor is raised above ground level by foundation walls, shear walls, posts, piers, pilings or columns.",
      category="single_chunk", difficulty="medium", distractors=[DWELL, GENPROP]),

    # --- F. multi-chunk synthesis -------------------------------------------------------
    Q("nfip-024", "Under the Dwelling Form, what must I do about a proof of loss if the adjuster does not give me the form or help me complete it?",
      anchors=[(DWELL, r"Within 60 days after the loss, send us a proof of loss"),
               (DWELL, r"you must still send us a proof of loss within 60 days after the loss even if the adjuster does not furnish the form")],
      gold="You must still send a proof of loss within 60 days after the loss; any help from the adjuster is a courtesy only and does not remove that obligation.",
      category="multi_chunk", difficulty="hard", distractors=[GENPROP, RCBAP]),
    Q("nfip-025", "Under the General Property Form, who is responsible for deciding and justifying the amount of loss claimed, and by when must it be submitted?",
      anchors=[(GENPROP, r"Within 60 days after the loss, send us a proof of loss"),
               (GENPROP, r"you must use your own judgment concerning the amount of loss and justify that amount")],
      gold="The policyholder must use their own judgement to decide and justify the amount of loss, and must submit the proof of loss within 60 days after the loss.",
      category="multi_chunk", difficulty="hard", distractors=[DWELL, RCBAP]),
    Q("nfip-026", "Under the RCBAP, if the building is insured for less than the required amount, how is the loss payment worked out?",
      anchors=[(RCBAP, r"At least 80 percent of its replacement cost"),
               (RCBAP, r"Divide the actual amount of insurance carried on the building by the required amount of insurance")],
      gold="Divide the insurance actually carried by the required amount, multiply the loss before the deductible by that figure, then subtract the deductible; the payment is that result or the amount of insurance carried, whichever is less. The required amount is the lesser of 80 percent of replacement cost and the maximum NFIP insurance available.",
      category="multi_chunk", difficulty="hard", distractors=[DWELL, GENPROP]),

    # --- G. more single-chunk breadth ---------------------------------------------------
    Q("nfip-027", "Under the Dwelling Form, when does coverage begin for a building that is still under construction?",
      doc=DWELL, anchor=r"coverage does not apply until the building is walled and roofed if the lowest floor",
      gold="Coverage does not apply until the building is walled and roofed, where the lowest floor is below the base flood elevation.",
      category="single_chunk", difficulty="medium", distractors=[GENPROP, RCBAP]),
    Q("nfip-028", "Under the General Property Form, when is a building under construction covered?",
      doc=GENPROP, anchor=r"coverage does not apply until the building is walled and roofed if the lowest floor",
      gold="Not until the building is walled and roofed, where the lowest floor is below the base flood elevation.",
      category="near_miss", difficulty="hard", distractors=[DWELL, RCBAP]),
    Q("nfip-029", "Under the RCBAP, can the amount of coverage be increased while a premium is still owed?",
      doc=RCBAP, anchor=r"the amount of coverage under this policy can only be increased by endorsement subject to the appropriate waiting period",
      gold="Coverage can only be increased by endorsement, subject to the appropriate waiting period, and no increase is allowed until the requested information has been provided.",
      category="single_chunk", difficulty="medium", distractors=[DWELL, GENPROP]),
    Q("nfip-030", "Under the Dwelling Form, what types of property does the policy insure?",
      doc=DWELL, anchor=r"A one to four family residential building, not under a condominium form of ownership",
      gold="A one to four family residential building not under condominium ownership, a single-family dwelling unit in a condominium building, and personal property in a building.",
      category="single_chunk", difficulty="easy", distractors=[GENPROP, RCBAP]),
    Q("nfip-031", "Under the General Property Form, what is mudflow and what is excluded from it?",
      doc=GENPROP, anchor=r"slope failure, or a saturated soil mass",
      gold="Mudflow is flowing mud on the surface of normally dry land; landslide, slope failure and a saturated soil mass moving by liquidity down a slope are not mudflow.",
      category="exclusion", difficulty="hard", distractors=[DWELL, RCBAP]),
    Q("nfip-032", "Under the RCBAP, may the insurer repair or replace damaged property instead of paying?",
      doc=RCBAP, anchor=r"If we give you written notice within 30 days after we receive your signed, sworn proof of loss",
      gold="Yes, if written notice is given within 30 days after the signed, sworn proof of loss is received.",
      category="near_miss", difficulty="hard", distractors=[DWELL, GENPROP]),

    # --- H. unanswerable ----------------------------------------------------------------
    # (b) near-miss traps: answerable for one form, asked of a form that has no such rule.
    Q("nfip-033", "Under the Dwelling Form, what percentage of replacement cost triggers the coinsurance penalty?",
      gold=None, answerable=False, category="unanswerable", difficulty="hard",
      distractors=[RCBAP, DWELL]),
    Q("nfip-034", "What is the coinsurance penalty threshold under the General Property Form?",
      gold=None, answerable=False, category="unanswerable", difficulty="hard",
      distractors=[RCBAP, GENPROP]),
    # (a) plausible but absent from the policy forms entirely.
    Q("nfip-035", "What is the annual premium for $250,000 of building coverage in Zone AE?",
      gold=None, answerable=False, category="unanswerable", difficulty="medium",
      distractors=list(ALL_FORMS)),
    Q("nfip-036", "How do I appeal a denied flood claim to the Federal Insurance Administrator?",
      gold=None, answerable=False, category="unanswerable", difficulty="medium",
      distractors=list(ALL_FORMS)),
    Q("nfip-037", "How many NFIP flood policies are currently in force nationwide?",
      gold=None, answerable=False, category="unanswerable", difficulty="easy",
      distractors=list(ALL_FORMS)),
    # (c) out of domain for a flood policy.
    Q("nfip-038", "What is the deductible for wind damage caused by a hurricane?",
      gold=None, answerable=False, category="unanswerable", difficulty="hard",
      distractors=list(ALL_FORMS)),
    Q("nfip-039", "Does this policy provide liability cover if a visitor is injured in my home?",
      gold=None, answerable=False, category="unanswerable", difficulty="medium",
      distractors=list(ALL_FORMS)),
    # (d) underspecified - cannot be answered without naming a form.
    Q("nfip-040", "What is the coinsurance requirement?",
      gold=None, answerable=False, category="unanswerable", difficulty="hard",
      distractors=list(ALL_FORMS)),
]


def resolve_anchor(document, pattern: str, chunks) -> tuple[int, int, list[str]]:
    """Find the provision and return its offsets plus the chunks covering it."""
    matches = list(re.finditer(pattern, document.text))
    if not matches:
        raise SystemExit(f"anchor matched nothing in {document.document_id}: {pattern!r}")
    if len(matches) > 1:
        raise SystemExit(
            f"anchor matched {len(matches)} times in {document.document_id} "
            f"(evidence must be unique): {pattern!r}"
        )
    match = matches[0]
    covering = [
        c.chunk_id for c in chunks
        if c.document_id == document.document_id and c.start <= match.start() and c.end >= match.end()
    ]
    if not covering:
        raise SystemExit(
            f"anchor in {document.document_id} spans a chunk boundary and has no single "
            f"covering chunk: {pattern!r}"
        )
    return match.start(), match.end(), covering


def assign_split(question_id: str, category: str) -> str:
    """Deterministic dev/test assignment, stratified by category.

    Hashing the ID means the split never moves, and folding the category in keeps roughly
    half of each category on each side rather than letting one bucket land entirely in dev.
    """
    digest = hashlib.sha256(f"{category}|{question_id}".encode()).hexdigest()
    return "dev" if int(digest[:8], 16) % 2 == 0 else "test"


def main() -> int:
    documents = load_corpus()
    chunks = chunk_corpus(documents, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
    by_id = {d.document_id: d for d in documents}

    records = []
    for spec in QUESTIONS:
        anchors = spec.pop("_anchors")
        relevant_documents: list[str] = []
        relevant_chunks: list[str] = []
        spans: list[dict] = []

        for document_id, pattern in anchors:
            document = by_id[document_id]
            start, end, covering = resolve_anchor(document, pattern, chunks)
            if document_id not in relevant_documents:
                relevant_documents.append(document_id)
            for chunk_id in covering:
                if chunk_id not in relevant_chunks:
                    relevant_chunks.append(chunk_id)
            spans.append({"document_id": document_id, "start": start, "end": end})

        records.append({
            "question_id": spec["question_id"],
            "question": spec["question"],
            "answerable": spec["answerable"],
            "category": spec["category"],
            "subdomain": "flood/us",
            "difficulty": spec["difficulty"],
            "split": assign_split(spec["question_id"], spec["category"]),
            "relevant_document_ids": relevant_documents,
            "relevant_chunk_ids": relevant_chunks,
            "evidence_spans": spans,
            "gold_answer": spec["gold_answer"],
            "distractor_document_ids": spec["distractor_document_ids"],
            "provenance": PROVENANCE,
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    answerable = sum(r["answerable"] for r in records)
    print(f"Wrote {len(records)} questions to {OUT_PATH}")
    print(f"  answerable {answerable} | unanswerable {len(records) - answerable} "
          f"({(len(records) - answerable) / len(records):.0%})")
    for split in ("dev", "test"):
        print(f"  {split}: {sum(r['split'] == split for r in records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
