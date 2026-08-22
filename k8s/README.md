# Phase 4 — Containerization & Kubernetes

## Part A — Docker (verified by hand, not in CI)

Build the API image and run it as a container. It talks to Qdrant + Ollama running on the
host (via `host.docker.internal`).

```bash
docker build -t insureassist:latest .

docker run -d --name insureassist-api -p 8010:8000 ^
  -e QDRANT_URL=http://host.docker.internal:6533 ^
  -e LLM_BACKEND=ollama ^
  -e OLLAMA_URL=http://host.docker.internal:11434 ^
  -e OLLAMA_MODEL=llama3.2:3b ^
  -e EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 ^
  insureassist:latest

# test:
curl http://localhost:8010/health
curl -X POST http://localhost:8010/ask -H "Content-Type: application/json" ^
  -d "{\"question\": \"Does home insurance cover a burst pipe?\"}"
```
Checked by hand: `/health` returns `{"status":"ok"}` and `/ask` returns an answer with
its source documents. This was a manual run, not a reproducible or CI-verified result.

Note that `/health` reports only that the process is up. It says nothing about Qdrant
or the generator, so it is a liveness check and nothing more.

## Part B — Kubernetes

The manifests in this folder deploy the system to a cluster:
- `qdrant.yaml`         — the vector DB (Deployment + Service)
- `configmap.yaml`      — non-secret config (URLs, model names)
- `api-deployment.yaml` — the RAG API (2 replicas, health probes, resource limits)
- `api-service.yaml`    — exposes the API (LoadBalancer)
- `hpa.yaml`            — autoscaling (2–6 pods on CPU load)

### Option 1 — Local cluster (Docker Desktop Kubernetes, easiest)
1. Docker Desktop → **Settings → Kubernetes → Enable Kubernetes → Apply & restart**.
2. Load the image and deploy:
   ```bash
   kubectl apply -f k8s/qdrant.yaml
   kubectl apply -f k8s/configmap.yaml
   kubectl apply -f k8s/api-deployment.yaml
   kubectl apply -f k8s/api-service.yaml
   kubectl get pods
   kubectl port-forward service/insureassist-api 8080:80
   # then: curl http://localhost:8080/health
   ```
   (For a purely local run, point the ConfigMap's `LLM_BACKEND` to `ollama` and
   `OLLAMA_URL` to `http://host.docker.internal:11434`.)

### Option 2 — Managed Kubernetes (GKE)
The same manifests are intended to deploy to Google Kubernetes Engine; see
`docs/gcp_deploy.md`. **This has not been done.** The guide is untested and the deploy
workflow has never completed successfully.

## What this demonstrates
- Containerised a Python service with a single-stage Dockerfile and ran it by hand.
- Authored Kubernetes manifests: Deployment, Service, ConfigMap, probes, resource
  limits, and a HorizontalPodAutoscaler.

## Known gaps in these manifests
- **Readiness uses `/health`**, which always returns ok. A pod is marked ready even if
  Qdrant or the generator is unreachable, so traffic can be routed to a pod that
  cannot answer.
- **Qdrant has no persistent volume.** It is a Deployment writing to the container
  filesystem, so a restart loses the whole index and the ingest Job must be re-run.
- **Ollama's model directory is an `emptyDir`**, so the model must be pulled again by
  hand after any restart.
- **No `securityContext`.** Combined with the root container image, pods run as root.
- The manifests pin `insureassist:latest` while the deploy workflow pushes a
  commit-tagged image; the two disagree.
