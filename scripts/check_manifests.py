"""
Assert the invariants the Kubernetes manifests must hold.

    python -m scripts.check_manifests

These are the properties that were actually wrong before, so they are checked rather than
trusted: readiness must not be liveness, nothing may run as root, Qdrant must have real
persistence, and no manifest may pin ":latest".
"""
from __future__ import annotations

import glob
import sys

import yaml


def load_all() -> list[dict]:
    documents = []
    for path in sorted(glob.glob("k8s/*.yaml")):
        with open(path, encoding="utf-8") as f:
            documents += [(path, d) for d in yaml.safe_load_all(f) if d]
    return documents


def main() -> int:
    problems = []
    documents = load_all()
    kinds = {d.get("kind") for _, d in documents}

    for path, doc in documents:
        kind = doc.get("kind")
        spec = doc.get("spec", {})
        template = spec.get("template", {}).get("spec", {})
        containers = template.get("containers", [])

        for container in containers:
            image = container.get("image", "")
            if image.endswith(":latest"):
                problems.append(f"{path}: {container['name']} pins ':latest'")

            if kind in ("Deployment", "StatefulSet"):
                liveness = container.get("livenessProbe", {}).get("httpGet", {}).get("path")
                readiness = container.get("readinessProbe", {}).get("httpGet", {}).get("path")
                if container["name"] == "api":
                    if liveness != "/health":
                        problems.append(f"{path}: api liveness should be /health, got {liveness}")
                    if readiness != "/ready":
                        problems.append(f"{path}: api readiness should be /ready, got {readiness}")
                    if liveness == readiness:
                        problems.append(f"{path}: liveness and readiness must differ")

        if kind in ("Deployment", "StatefulSet", "Job") and containers:
            security = template.get("securityContext", {})
            if not security.get("runAsNonRoot"):
                problems.append(f"{path}: pod securityContext must set runAsNonRoot")

    # Qdrant must not be a Deployment writing to the container filesystem.
    qdrant = [d for _, d in documents if d.get("metadata", {}).get("name") == "qdrant"]
    stateful = [d for d in qdrant if d.get("kind") == "StatefulSet"]
    if not stateful:
        problems.append("k8s: qdrant must be a StatefulSet with persistent storage")
    else:
        if not stateful[0]["spec"].get("volumeClaimTemplates"):
            problems.append("k8s: qdrant StatefulSet has no volumeClaimTemplates")

    if "Deployment" not in kinds:
        problems.append("k8s: no Deployment found")

    if problems:
        print(f"FAIL: {len(problems)} manifest problem(s)\n")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print(f"Manifest invariants hold ({len(documents)} documents checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
