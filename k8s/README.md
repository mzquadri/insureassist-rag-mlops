# Phase 4 — Containerization & Kubernetes

## Part A — Docker (verified working locally)

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
Verified: `/health` returns `{"status":"ok"}` and `/ask` returns a grounded answer with
sources — the whole RAG service runs inside a container.

## Part B — Kubernetes

The manifests in this folder deploy the system to a cluster:
- `qdrant.yaml`         — the vector DB (Deployment + Service)
- `configmap.yaml`      — non-secret config (URLs, model names)
- `secret.example.yaml` — template for API keys (copy to `secret.yaml`)
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

### Option 2 — Real managed Kubernetes (GKE) — Phase 5
The same manifests deploy to Google Kubernetes Engine. See `docs/gcp_deploy.md`.
This is the more impressive, production-grade path and is covered in Phase 5.

## What you learn / can claim
- Containerized a Python service with a multi-stage Dockerfile (verified).
- Authored Kubernetes manifests: Deployment, Service, ConfigMap, Secret, health probes,
  resource limits, and HorizontalPodAutoscaler.
