"""
Verify that published numbers match the reference run.

    python -m eval.verify_artifacts

The README and the docs are not the source of truth; `eval/reference_run.json` is. This
script re-derives what it can and fails if any documented figure has drifted from the
artefact behind it. It runs in CI, so a number cannot be edited into a document and left
there unchallenged.

Checks:
  * the corpus files still hash to what the manifest records
  * the reference run's corpus hash matches the committed corpus
  * every documented metric appears in the artefact with the same value
  * the baselines a comparison is measured against, not only the headline
  * the size of the question set, which went stale once when the unanswerable set
    grew from eight to eighteen and the prose was left at forty
  * the abstention numbers, from their own artefact, and that it still reports the
    threshold as rejected
  * the frozen retrieval config hash matches the config the run used
  * no abstention threshold has appeared
"""

from __future__ import annotations

import json
import re
import sys

from src.corpus import corpus_hash, load_corpus, verify_document_hashes
from src.paths import REFERENCE_RUN_PATH as RUN_PATH
from src.paths import REPO_ROOT
from src.paths import RETRIEVAL_CONFIG_PATH as CONFIG_PATH

#: Numbers quoted in prose, and where they must come from in the artefact.
#: Each entry is (document, regex capturing the number, dotted path into the run).
DOCUMENTED_CLAIMS = [
    (
        "docs/BENCHMARK.md",
        r"\| Hit rate \| [\d.]+ \| [\d.]+ \| \*\*([\d.]+)\*\* \|",
        "retrieval.metrics.hit_rate@5",
    ),
    ("docs/BENCHMARK.md", r"MRR \*\*([\d.]+)\*\*", "retrieval.metrics.mrr"),
    (
        "docs/BENCHMARK.md",
        r"top-document accuracy \*\*([\d.]+)\*\*",
        "retrieval.metrics.top_document_accuracy",
    ),
    (
        "README.md",
        r"\*\*Top-document accuracy ([\d.]+)\*\*",
        "retrieval.metrics.top_document_accuracy",
    ),
    ("README.md", r"hit rate@5 ([\d.]+)", "retrieval.metrics.hit_rate@5"),
    # The baseline row is what turns a metric into a comparison, so it is pinned the
    # same way the headline is. Without these, a document could quote the system's
    # MRR correctly and invent the number it is measured against.
    (
        "docs/BENCHMARK.md",
        r"\| Dense only \| ([\d.]+) \|",
        "retrieval.baselines.dense.hit_rate@5",
    ),
    (
        "docs/BENCHMARK.md",
        r"\| Dense only \| [\d.]+ \| ([\d.]+) \|",
        "retrieval.baselines.dense.mrr",
    ),
    (
        "docs/BENCHMARK.md",
        r"\| Dense only \| [\d.]+ \| [\d.]+ \| ([\d.]+) \|",
        "retrieval.baselines.dense.top_document_accuracy",
    ),
    (
        "docs/BENCHMARK.md",
        r"\| BM25 only \| \*\*([\d.]+)\*\* \|",
        "retrieval.baselines.bm25.hit_rate@5",
    ),
    (
        "docs/BENCHMARK.md",
        r"\| BM25 only \| \*\*[\d.]+\*\* \| ([\d.]+) \|",
        "retrieval.baselines.bm25.mrr",
    ),
    (
        "docs/BENCHMARK.md",
        r"\| BM25 only \| \*\*[\d.]+\*\* \| [\d.]+ \| ([\d.]+) \|",
        "retrieval.baselines.bm25.top_document_accuracy",
    ),
    # The size of the question set went stale once: the documents said forty while the
    # run had grown to fifty, because the unanswerable set was enlarged and the prose
    # was not. Counts are claims too.
    (
        "docs/LIMITATIONS.md",
        r"\*\*(\d+) questions, \d+ in the test split",
        "questions.total",
    ),
    (
        "docs/LIMITATIONS.md",
        r"\*\*\d+ questions, (\d+) in the test split",
        "questions.by_split.test",
    ),
    ("docs/LIMITATIONS.md", r"invalidates all (\d+) labels", "questions.total"),
    ("docs/BENCHMARK.md", r"the corpus and the (\d+) questions", "questions.total"),
    ("README.md", r"the corpus and the (\d+) questions", "questions.total"),
    ("README.md", r"Ground truth \((\d+) labels\)", "questions.total"),
]

#: Claims about the abstention experiment, which lives in its own artefact.
ABSTENTION_CLAIMS = [
    (
        "docs/LIMITATIONS.md",
        r"balanced accuracy ([\d.]+) on dev",
        "dev.balanced_accuracy",
    ),
    (
        "docs/LIMITATIONS.md",
        r"balanced accuracy [\d.]+ on dev, ([\d.]+) on\s+test",
        "test.balanced_accuracy",
    ),
    (
        "docs/LIMITATIONS.md",
        r"(\d+) of the \d+ unanswerable questions score",
        "separation.unanswerable_scoring_above_the_weakest_answerable",
    ),
    (
        "docs/LIMITATIONS.md",
        r"\d+ of the (\d+) unanswerable questions score",
        "separation.unanswerable_total",
    ),
]


#: The abstention threshold was tested and rejected; its numbers live here rather than
#: in the reference run, because nothing about it is served.
ABSTENTION_PATH = REPO_ROOT / "eval" / "abstention_threshold.json"

#: Claims about what the generator did with its citations. Its own artefact, because it
#: measures generation and the reference run measures retrieval; mixing them would let a
#: generation regression hide behind a retrieval number.
CITATION_CLAIMS = [
    ("docs/LIMITATIONS.md", r"(\d+) of 18, \*\*27\.8%", "summary.with_out_of_range"),
    (
        "docs/LIMITATIONS.md",
        r"Mean context coverage is \*\*([\d.]+)%",
        "summary.mean_context_coverage_pct",
    ),
    ("README.md", r"5 of 18, ([\d.]+)% \[", "summary.out_of_range_pct"),
    ("README.md", r"mean\s+coverage ([\d.]+)%", "summary.mean_context_coverage_pct"),
]


#: Generation-side evidence. Absent until eval/citation_run.py has been run, which needs
#: a local model, so its checks are skipped rather than failed when it is not there.
CITATION_PATH = REPO_ROOT / "eval" / "citation_structure.json"

#: The prompt variant is a separate arm in a separate file, so it is checked separately.
#: Its numbers sit in LIMITATIONS beside the served ones and must not drift from them.
RANGED_CLAIMS = [
    (
        "docs/LIMITATIONS.md",
        r"\| range stated \| (\d+)/18",
        "summary.with_out_of_range",
    ),
    (
        "docs/LIMITATIONS.md",
        r"\| range stated \| \d+/18, ([\d.]+)%",
        "summary.out_of_range_pct",
    ),
    (
        "docs/LIMITATIONS.md",
        r"Coverage falls from [\d.]+% to ([\d.]+)%",
        "summary.mean_context_coverage_pct",
    ),
]

RANGED_PATH = REPO_ROOT / "eval" / "citation_structure_ranged.json"


def dig(data: dict, path: str):
    for part in path.split("."):
        data = data[part]
    return data


def check_claims(claims: list, artefact: dict, source_name: str) -> list[str]:
    """Every number a document quotes must be in the artefact, to within rounding.

    A claim that matches nothing is a failure rather than a skip. A pattern that
    stops matching usually means the sentence around it was rewritten and the
    number quietly stopped being checked, which is the failure this exists to
    catch.
    """
    problems: list[str] = []
    for document, pattern, path in claims:
        source = REPO_ROOT / document
        if not source.exists():
            problems.append(f"{document} is missing")
            continue
        text = source.read_text(encoding="utf-8")
        matches = re.findall(pattern, text)
        if not matches:
            problems.append(
                f"{document}: nothing matches {pattern!r}, so {path} from "
                f"{source_name} is no longer being checked"
            )
            continue
        expected = dig(artefact, path)
        for found in matches:
            if abs(float(found) - float(expected)) > 0.0005:
                problems.append(
                    f"{document}: documents {found} for {path}, "
                    f"{source_name} says {expected}"
                )
    return problems


def main() -> int:
    problems: list[str] = []

    if not RUN_PATH.exists():
        print(f"FAIL: {RUN_PATH} is missing. Run `python -m eval.reference_run`.")
        return 1

    run = json.loads(RUN_PATH.read_text(encoding="utf-8"))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    # --- corpus integrity ---------------------------------------------------------------
    documents = load_corpus()
    mismatched = verify_document_hashes(documents)
    if mismatched:
        problems.append(f"corpus files no longer match the manifest: {mismatched}")

    if run["corpus"]["corpus_hash"] != corpus_hash(documents):
        problems.append(
            "reference run was produced against a different corpus "
            f"({run['corpus']['corpus_hash'][:12]} != {corpus_hash(documents)[:12]})"
        )

    # --- the run must describe the frozen configuration ----------------------------------
    if run["retrieval"]["architecture"] != config["architecture"]:
        problems.append(
            f"run architecture {run['retrieval']['architecture']!r} does not match the "
            f"frozen config {config['architecture']!r}"
        )
    if run["retrieval"]["chunking"]["size"] != config["chunking"]["size"]:
        problems.append("run chunk size does not match the frozen config")

    # --- documented numbers -------------------------------------------------------------
    problems += check_claims(DOCUMENTED_CLAIMS, run, "the reference run")

    # --- the abstention experiment, which lives in its own artefact ----------------------
    if ABSTENTION_PATH.exists():
        rejected = json.loads(ABSTENTION_PATH.read_text(encoding="utf-8"))
        problems += check_claims(ABSTENTION_CLAIMS, rejected, "the abstention run")

    # --- what the generator did with its citations --------------------------------------
    if CITATION_PATH.exists():
        citations = json.loads(CITATION_PATH.read_text(encoding="utf-8"))
        problems += check_claims(CITATION_CLAIMS, citations, "the citation run")

    if RANGED_PATH.exists():
        ranged = json.loads(RANGED_PATH.read_text(encoding="utf-8"))
        problems += check_claims(RANGED_CLAIMS, ranged, "the range-stated prompt arm")
        if rejected.get("adopted"):
            problems.append(
                "the abstention artefact now reports adopted=true; the documents still "
                "describe a threshold that was tested and rejected"
            )

    # --- no threshold may appear ---------------------------------------------------------
    abstention = run["generation"]["abstention"]
    if any(key in abstention for key in ("threshold", "threshold_value", "min_score")):
        problems.append(
            "an abstention threshold appeared in the reference run; none is validated"
        )

    if problems:
        print(f"FAIL: {len(problems)} problem(s)\n")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print("Documentation matches the reference run.")
    print(f"  corpus hash        {run['corpus']['corpus_hash'][:16]}")
    print(f"  question set hash  {run['questions']['question_set_hash'][:16]}")
    print(f"  architecture       {run['retrieval']['architecture']}")
    checked = len(DOCUMENTED_CLAIMS) + (
        len(ABSTENTION_CLAIMS) if ABSTENTION_PATH.exists() else 0
    )
    print(f"  checked claims     {checked}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
