"""
Compare two reference runs and fail if the deterministic parts drifted.

    python -m eval.compare_runs eval/reference_run.json /tmp/ci_reference_run.json

Retrieval is deterministic, so a committed run and a freshly produced one must agree on
every retrieval metric and every hash. Latency is excluded: it is machine-specific.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TOLERANCE = 1e-6


def main(committed: str, produced: str) -> int:
    a = json.loads(Path(committed).read_text(encoding="utf-8"))
    b = json.loads(Path(produced).read_text(encoding="utf-8"))
    problems = []

    for path in ("corpus.corpus_hash", "questions.question_set_hash",
                 "reproducibility.retrieval_config_hash", "retrieval.architecture"):
        va, vb = a, b
        for part in path.split("."):
            va, vb = va[part], vb[part]
        if va != vb:
            problems.append(f"{path}: committed {va} != produced {vb}")

    for name, value in a["retrieval"]["metrics"].items():
        other = b["retrieval"]["metrics"].get(name)
        if other is None or abs(value - other) > TOLERANCE:
            problems.append(f"retrieval.{name}: committed {value} != produced {other}")

    for baseline, metrics in a["retrieval"]["baselines"].items():
        for name, value in metrics.items():
            other = b["retrieval"]["baselines"].get(baseline, {}).get(name)
            if other is None or abs(value - other) > TOLERANCE:
                problems.append(f"baseline {baseline}.{name}: {value} != {other}")

    for name in ("citation_precision", "citation_recall", "unsupported_citation_rate"):
        va = a["generation"]["citations"][name]
        vb = b["generation"]["citations"][name]
        if abs(va - vb) > TOLERANCE:
            problems.append(f"citations.{name}: committed {va} != produced {vb}")

    if problems:
        print(f"FAIL: {len(problems)} drift(s) between committed and reproduced run\n")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print("Reproduced run matches the committed reference run.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
