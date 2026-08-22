# Roadmap — build it phase by phase

Each phase is self-contained and adds one capability to the project. Do them in order;
every phase ends with something you can run and verify. No prior experience with these
tools is assumed — each phase explains the idea first, then the commands.

Legend:  🎯 the skill it teaches   ✔ what you can run at the end

---

## Phase 0 — Setup (~30 min)  🎯 tooling & accounts

1. Install: [Python 3.11+](https://www.python.org/downloads/),
   [Docker Desktop](https://www.docker.com/products/docker-desktop/),
   [Git](https://git-scm.com/download/win).
2. Create free accounts (only needed for later phases): GitHub, Google Cloud
   (for GKE), Hugging Face, and — optionally — LangSmith.
3. Set up the environment:
   ```bash
   python -m venv .venv && .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   copy .env.example .env
   ```

---

## Phase 1 — Local RAG baseline (~½ day)  🎯 embeddings, vector search, serving

A working RAG API on your machine, no cloud required.

```bash
docker compose up -d qdrant
ollama pull llama3.2:3b
python -m src.ingest
uvicorn src.api:app --reload
```

✔ Ask questions at http://localhost:8000/docs → `/ask`. You now have retrieval
(BGE embeddings + Qdrant) feeding an LLM, served through FastAPI.

---

## Phase 2 — LoRA fine-tuning + MLflow (~1 day)  🎯 LLM fine-tuning, experiment tracking

Instead of training a whole model, **LoRA** trains a tiny adapter (a few MB) — cheap enough
to run on a free Colab GPU.

- Open `archive/finetune/lora_finetune.ipynb` in Google Colab, set the runtime to a **T4 GPU**,
  and run all cells. It fine-tunes Phi-3-mini with Hugging Face **PEFT** and logs the run
  to **MLflow**.
- Download the resulting `adapter/` folder.

✔ *Intended outcome:* a LoRA adapter for the insurance domain.

> Not achieved in this repository. The notebook is committed with no outputs, no
> adapter was produced, and the served system uses a stock model. The training pairs
> are also the evaluation questions, so a model trained this way could not be scored
> honestly. See `archive/finetune/README.md`.

---

## Phase 3 — Evaluation (~½ day)  🎯 RAG evaluation

You can't ship what you haven't measured. `eval/evaluate.py` scores the RAG pipeline with
an **LLM-as-judge** on four metrics: faithfulness, answer relevancy, context precision,
and answer correctness.

```bash
python -m eval.evaluate
```

✔ A metrics table and `eval/eval_report.csv` you can track over time.

---

## Phase 4 — Docker & Kubernetes (~½ day)  🎯 containerization, orchestration

```bash
docker build -t insureassist:latest .
docker run -d -p 8010:8000 --env-file .env insureassist:latest   # (see k8s/README.md)
```

Then deploy to a cluster with the manifests in `k8s/` (Deployment, Service, ConfigMap,
HorizontalPodAutoscaler). See [`k8s/README.md`](k8s/README.md).

✔ The service running as a container (verified by hand), and Kubernetes manifests
written. The manifests have not been applied to a cluster.

---

## Phase 5 — Google Cloud (GKE) (~1 day)  🎯 managed Kubernetes in the cloud

Push the image to Artifact Registry and deploy to a managed **GKE** cluster. Full
steps in [`docs/gcp_deploy.md`](docs/gcp_deploy.md).

✔ *Intended outcome:* a publicly reachable endpoint on GKE.

> Not achieved. The guide has not been executed and the deploy workflow has never
> succeeded. Cloud Storage is not used by any code in this repository.

---

## Phase 6 — CI/CD (~½ day)  🎯 automation

`.github/workflows/ci.yml` runs checks on every push. `.github/workflows/deploy.yml`
builds and deploys to GKE (add your GCP secrets to enable it). Add an autoscaler
(`k8s/hpa.yaml`) so pods scale with load.

✔ Push code → lint, static validation and the offline test suite run automatically.

> CI does not build the image, start Qdrant, or make an HTTP request, so a green tick
> means "parses, lints, and passes offline tests" - not "the RAG pipeline works".
> The deploy workflow is manual-only and unproven.
