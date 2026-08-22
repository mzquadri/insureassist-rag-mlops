# Production boundary

What is real, what is authored, and what has never been run.

## Real, and verified in CI

**API semantics.** `/health` is liveness and touches no dependency. `/ready` probes the
embedder, the lexical index and the vector store, returning 503 with per-dependency status.
Splitting them fixed a genuine defect: one `/health` serving both roles meant Kubernetes
marked pods ready while Qdrant was unreachable, then routed traffic to something that could
only fail.

**Failure handling.** Missing dependencies are 503 with a machine-readable `dependency`
field, never 500. A configuration typo stays a 500-class error because retrying will not fix
it. Internal exceptions never reach the caller, and every response carries a request ID
(honoured from `X-Request-ID` if the caller supplies one).

**Container.** Genuinely two-stage. Runs as uid 10001, non-root, with a `HEALTHCHECK` and
fully pinned dependencies. CI builds it, asserts the uid, starts it against a real Qdrant,
runs ingestion through the same image, and exercises live HTTP.

**Ingestion.** Deterministic and idempotent - re-ingesting leaves exactly 314 points. CI runs
it twice and checks the count.

## Authored, never deployed

**Kubernetes manifests.** Parsed and invariant-checked in CI: `scripts/check_manifests.py`
enforces liveness not equal to readiness, non-root pod security contexts, no `:latest` tags,
and Qdrant persistence. They have **never been applied to a cluster**.

Fixed in this pass:

- Qdrant became a **StatefulSet with a PVC**. As a Deployment writing to the container
  filesystem, any restart destroyed the whole index and the ingest Job had to be re-run.
- Probes split: liveness on `/health`, readiness on `/ready`.
- `securityContext` added throughout: `runAsNonRoot`, dropped capabilities, no privilege
  escalation, read-only root filesystem for the API.
- `:latest` replaced with a pinned tag, matching what the deploy workflow pushes.
- A startup probe added so the first-use model download cannot be killed by liveness.
- Ollama model directory moved from `emptyDir` to a PVC.

**Ollama model provisioning is not automated.** The model must be pulled by hand once the pod
starts. The PVC means that pull survives restarts. Automating a roughly 2 GB pull is outside
scope for this project and is not pretended to exist.

**GKE.** `gcp_deploy.md` is untested guidance. The deploy workflow has never succeeded.

## Not implemented, on purpose

No rate limiting, service mesh, ingress controller, PodDisruptionBudget or NetworkPolicy.
None is needed by a single-service reference project, and adding them would be decoration.

The HPA exists and is unverified; there is no horizontal scaling evidence.

## What CI does not prove

CI runs no language model, so generation quality is not measured there. The container test
proves the request reaches the generator and fails at that boundary, which is the honest
limit of what can be verified without shipping a 2 GB model into every Actions run.
