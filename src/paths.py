"""Repository-anchored paths.

Committed artefacts - the corpus, the frozen retrieval config, the ground truth, the
reference run - live at fixed locations inside the repository. Addressing them with
relative paths silently ties every reader to the process working directory: importing
`src.corpus` from anywhere other than the repository root raised `FileNotFoundError`,
which meant the API could only be started from one place.

`REPO_ROOT` is derived from this file's own location, so the paths resolve the same way
regardless of where the process was launched from.
"""
from __future__ import annotations

from pathlib import Path

#: The repository root - the parent of the ``src`` package this module lives in.
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"

EVAL_DIR = REPO_ROOT / "eval"
RETRIEVAL_CONFIG_PATH = EVAL_DIR / "retrieval_config.json"
REFERENCE_RUN_PATH = EVAL_DIR / "reference_run.json"
GROUND_TRUTH_PATH = EVAL_DIR / "ground_truth" / "nfip_questions.jsonl"
