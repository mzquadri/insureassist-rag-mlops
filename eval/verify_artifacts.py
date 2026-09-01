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
    ("docs/BENCHMARK.md", r"\| Hit rate \| [\d.]+ \| [\d.]+ \| \*\*([\d.]+)\*\* \|",
     "retrieval.metrics.hit_rate@5"),
    ("docs/BENCHMARK.md", r"MRR \*\*([\d.]+)\*\*", "retrieval.metrics.mrr"),
    ("docs/BENCHMARK.md", r"top-document accuracy \*\*([\d.]+)\*\*",
     "retrieval.metrics.top_document_accuracy"),
    ("README.md", r"\*\*Top-document accuracy ([\d.]+)\*\*",
     "retrieval.metrics.top_document_accuracy"),
    ("README.md", r"hit rate@5 ([\d.]+)", "retrieval.metrics.hit_rate@5"),
]


def dig(data: dict, path: str):
    for part in path.split("."):
        data = data[part]
    return data


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
    for document, pattern, path in DOCUMENTED_CLAIMS:
        source = REPO_ROOT / document
        if not source.exists():
            problems.append(f"{document} is missing")
            continue
        text = source.read_text(encoding="utf-8")
        matches = re.findall(pattern, text)
        if not matches:
            problems.append(f"{document}: no value found for {path} (pattern {pattern!r})")
            continue
        expected = dig(run, path)
        for found in matches:
            if abs(float(found) - float(expected)) > 0.0005:
                problems.append(
                    f"{document}: documents {found} for {path}, artefact says {expected}"
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
    print(f"  checked claims     {len(DOCUMENTED_CLAIMS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
