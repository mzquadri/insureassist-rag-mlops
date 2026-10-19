"""Export the chunking sweep's MLflow runs to a tracked JSON record.

`eval/chunking_sweep.py --mlflow` logs a parent run with one nested child per
chunk configuration. That tracking store is a SQLite file, which is gitignored
and not reviewable in a diff, so this writes the run structure out as JSON.

The point of the record is the shape, not the metrics: the metrics are already in
`eval/dev_chunking_sweep_reproduced.json`. What this adds is evidence that the
parent/child run layout the sweep documents is what it actually produced.

    MLFLOW_TRACKING_URI=sqlite:///mlflow.db python -m eval.export_mlflow_sweep

Refuses to write when the experiment holds no runs, so an empty export cannot be
mistaken for evidence of a run.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

EXPERIMENT = "insureassist-chunking"
OUT = Path(__file__).resolve().parent / "mlflow_chunking_runs.json"


def main() -> int:
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print("MLFLOW_TRACKING_URI is not set", file=sys.stderr)
        return 2

    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        print(f"no experiment named {EXPERIMENT!r} in {uri}", file=sys.stderr)
        return 1

    runs = client.search_runs([experiment.experiment_id])
    if not runs:
        print("the experiment holds no runs; refusing to write an empty record", file=sys.stderr)
        return 1

    def describe(run):
        return {
            "run_name": run.data.tags.get("mlflow.runName"),
            "status": run.info.status,
            "is_nested": bool(run.data.tags.get("mlflow.parentRunId")),
            "params": dict(sorted(run.data.params.items())),
            "metrics": {k: v for k, v in sorted(run.data.metrics.items())},
        }

    parents = [r for r in runs if not r.data.tags.get("mlflow.parentRunId")]
    children = sorted(
        (r for r in runs if r.data.tags.get("mlflow.parentRunId")),
        key=lambda r: (r.data.metrics.get("chunks", 0)),
    )

    record = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {
            "python": ".".join(str(n) for n in sys.version_info[:3]),
            "mlflow": mlflow.__version__,
            "tracking_store": uri.split("://", 1)[0],
        },
        "experiment": EXPERIMENT,
        "run_counts": {"parent": len(parents), "nested": len(children)},
        "parent_runs": [describe(r) for r in parents],
        "nested_runs": [describe(r) for r in children],
    }

    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name}: {len(parents)} parent run(s), {len(children)} nested run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
