"""Check that a candidate unanswerable question is one the corpus is actually silent on.

The distinction this enforces is easy to get wrong and expensive to get wrong. A
question about something the policy *explicitly excludes* is answerable: the form
says "we do not cover this", and a retrieval system is supposed to find that
sentence. A question is only unanswerable when the forms say nothing either way.

So each candidate declares the terms that would have to appear somewhere in the
corpus for the question to be answerable. If any of them do appear, the candidate
is rejected and has to be rewritten or dropped. Nothing is added to the ground
truth on the strength of an opinion about what the forms probably contain.

    python eval/check_unanswerable.py            # report
    python eval/check_unanswerable.py --emit     # print JSONL for the accepted ones
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
GROUND_TRUTH = ROOT / "eval" / "ground_truth" / "nfip_questions.jsonl"

PROVENANCE = "hand-authored against 44 CFR Pt.61 App. A(1)-A(3), silence verified by eval/check_unanswerable.py"


@dataclass(frozen=True)
class Candidate:
    """One proposed unanswerable question and the evidence that it is one."""

    question: str
    difficulty: str
    split: str
    # Terms that would have to be somewhere in the corpus for this to be
    # answerable. Any hit disqualifies the candidate.
    forbidden: tuple[str, ...]
    # Why the forms are silent, in one line, recorded in the review output.
    reason: str
    distractors: tuple[str, ...] = ("nfip-sfip-dwelling", "nfip-sfip-general-property")
    hits: list[str] = field(default_factory=list)


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        question="What premium would I pay for $250,000 of building coverage on a pre-FIRM home?",
        difficulty="medium",
        split="test",
        forbidden=("premium rate", "rate table", "rating table", "annual premium"),
        reason="The policy forms set out coverage and conditions; rating lives in the Flood Insurance Manual.",
    ),
    Candidate(
        question="Do I need an elevation certificate before this policy will pay a claim?",
        difficulty="medium",
        split="test",
        forbidden=("elevation certificate",),
        reason="The elevation certificate is an underwriting and rating document, not a condition of the contract.",
    ),
    Candidate(
        question="How much does a Community Rating System class 5 rating reduce what I pay?",
        difficulty="easy",
        split="dev",
        forbidden=("community rating system", "crs "),
        reason="CRS is a community programme administered outside the policy contract.",
    ),
    Candidate(
        question="Is my lender allowed to require me to carry this policy?",
        difficulty="medium",
        split="test",
        forbidden=("mandatory purchase", "required to purchase", "lender must"),
        reason="The mandatory purchase requirement sits in the Flood Disaster Protection Act, not in the SFIP.",
    ),
    Candidate(
        question="If the flood map changes, can I keep my old rating under grandfathering?",
        difficulty="hard",
        split="dev",
        forbidden=("grandfather",),
        reason="Grandfathering is a rating practice; the forms do not describe it.",
    ),
    Candidate(
        question="What makes a property a repetitive loss property?",
        difficulty="medium",
        split="test",
        forbidden=("repetitive loss", "severe repetitive"),
        reason="Repetitive loss is a mitigation programme designation defined outside the policy.",
    ),
    Candidate(
        question="How many days do I have to appeal a claim decision to FEMA?",
        difficulty="hard",
        split="test",
        forbidden=("appeal to fema", "appeals process", "may appeal"),
        reason="The appeals process is in 44 CFR Part 62, not in the policy form.",
    ),
    Candidate(
        question="Can I buy a Letter of Map Amendment to have my property removed from the flood zone?",
        difficulty="medium",
        split="dev",
        forbidden=("letter of map amendment", "loma", "map amendment"),
        reason="Map amendments are a FEMA mapping process with no mention in the contract.",
    ),
    Candidate(
        question="What commission does my insurance agent earn on this policy?",
        difficulty="easy",
        split="test",
        forbidden=("commission",),
        reason="Producer compensation is not part of the insured's contract.",
    ),
    Candidate(
        question="Am I eligible for a Group Flood Insurance Policy after a disaster declaration?",
        difficulty="hard",
        split="dev",
        forbidden=("group flood insurance", "group policy"),
        reason="The Group Flood Insurance Policy is a separate instrument issued under Part 61.",
    ),
    Candidate(
        question="Which deductible amounts can I choose from when I buy this policy?",
        difficulty="hard",
        split="test",
        forbidden=("deductible options", "schedule of deductibles", "available deductibles"),
        reason="The form applies whatever deductible is on the declarations; the menu of options is a rating matter.",
    ),
    Candidate(
        question="How does FEMA decide which adjuster is assigned to my claim?",
        difficulty="medium",
        split="dev",
        forbidden=("assign an adjuster", "adjuster assignment", "assigned adjuster"),
        reason="Adjuster assignment is an operational process of the insurer and FEMA, not a policy term.",
    ),
    Candidate(
        question="What grant funding is available to elevate my house after a flood?",
        difficulty="medium",
        split="test",
        forbidden=("grant", "hazard mitigation assistance"),
        reason="Mitigation grants are separate federal programmes; the policy pays claims, not grants.",
    ),
    Candidate(
        question="How long does the NFIP take on average to pay a claim?",
        difficulty="easy",
        split="dev",
        forbidden=("average", "processing time", "typically paid"),
        reason="Programme performance statistics are not contractual terms.",
    ),
    Candidate(
        question="Does my community have to join the NFIP for me to buy this policy?",
        difficulty="hard",
        split="test",
        forbidden=("participating community", "community participation", "eligible community"),
        reason="Community eligibility is a programme condition in Part 59, upstream of the contract.",
    ),
    Candidate(
        question="What is the coinsurance penalty if I insure a condominium building to less than 80 percent of replacement cost?",
        difficulty="hard",
        split="dev",
        forbidden=("80 percent", "80%", "coinsurance penalty"),
        reason=(
            "RCBAP does carry a coinsurance condition, so this one is on the edge. "
            "It is only admissible if the specific penalty arithmetic is absent."
        ),
        distractors=("nfip-sfip-rcbap",),
    ),
)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text)


def load_corpus() -> dict[str, str]:
    return {p.stem: normalise(p.read_text(encoding="utf-8", errors="replace"))
            for p in sorted(CORPUS.glob("*.txt"))}


def next_question_id(rows: list[dict]) -> int:
    highest = 0
    for row in rows:
        m = re.match(r"nfip-(\d+)$", row.get("question_id", ""))
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true", help="print JSONL for accepted candidates")
    args = parser.parse_args()

    corpus = load_corpus()
    if not corpus:
        print("no corpus found; run scripts/fetch_nfip_corpus.py first", file=sys.stderr)
        return 2

    rows = [json.loads(line) for line in GROUND_TRUTH.read_text(encoding="utf-8").splitlines() if line.strip()]
    existing = {normalise(r["question"]) for r in rows}

    accepted: list[Candidate] = []
    rejected: list[Candidate] = []

    for candidate in CANDIDATES:
        if normalise(candidate.question) in existing:
            candidate.hits.append("duplicate of an existing question")
            rejected.append(candidate)
            continue
        for term in candidate.forbidden:
            for doc, text in corpus.items():
                if normalise(term) in text:
                    candidate.hits.append(f"{doc} contains {term!r}")
        (rejected if candidate.hits else accepted).append(candidate)

    print(f"{len(CANDIDATES)} candidates, {len(accepted)} accepted, {len(rejected)} rejected\n")
    for candidate in rejected:
        print(f"REJECTED  {candidate.question}")
        for hit in candidate.hits:
            print(f"          {hit}")
        print()
    for candidate in accepted:
        print(f"accepted  [{candidate.split}/{candidate.difficulty}] {candidate.question}")

    if args.emit:
        print("\n--- JSONL ---")
        n = next_question_id(rows)
        for candidate in accepted:
            print(json.dumps({
                "question_id": f"nfip-{n:03d}",
                "question": candidate.question,
                "answerable": False,
                "category": "unanswerable",
                "subdomain": "flood/us",
                "difficulty": candidate.difficulty,
                "split": candidate.split,
                "relevant_document_ids": [],
                "relevant_chunk_ids": [],
                "evidence_spans": [],
                "gold_answer": None,
                "distractor_document_ids": list(candidate.distractors),
                "provenance": PROVENANCE,
            }))
            n += 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
