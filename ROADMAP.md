# Learning Roadmap — build it phase by phase

Each phase is self-contained, teaches one gap, and ends with **one strong CV bullet**.
Do them in order. Estimated total: ~2–3 weekends. You do NOT need to know any of this
beforehand — each phase explains the concept, then the commands.

Legend:  🎯 = the skill you learn   ✅ = the CV bullet you earn

---

## Phase 0 — Setup (30–45 min)  🎯 tooling, accounts

1. Install tools (Windows):
   - [Python 3.11+](https://www.python.org/downloads/) — check: `python --version`
   - [Docker Desktop](https://www.docker.com/products/docker-desktop/) (you already know Docker)
   - [Git](https://git-scm.com/download/win)
2. Create **free accounts** (all free tier):
   - GitHub (you have this: github.com/mzquadri)
   - Google Cloud — https://cloud.google.com/free ($300 free credit, 90 days) → for Phase 5
   - Hugging Face — https://huggingface.co/join → for models + optional adapter hosting
   - LangSmith — https://smith.langchain.com → free tracing (for Phase 3)
3. Clone/prepare this repo, create a virtual environment:
   ```powershell
   cd "C:\Users\MohdZaminQuadri\Downloads\insurance-rag-mlops"
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill keys as you get them (leave blank for now).

---

## Phase 1 — Local RAG baseline (½ day)  🎯 RAG you already know, cleanly structured

Goal: a working RAG API on your laptop, no cloud yet.
- Start Qdrant with Docker: `docker compose up -d qdrant`
- Ingest sample docs:   `python src/ingest.py`
- Run the API:          `uvicorn src.api:app --reload`
- Ask a question:       open http://localhost:8000/docs and try `/ask`

What you learn: embeddings (BGE via sentence-transformers), vector search (Qdrant),
retrieve-then-generate, FastAPI serving. (Base LLM for now; we fine-tune in Phase 2.)

✅ *"Built a production-structured RAG service (FastAPI + Qdrant + BGE embeddings) for
insurance-policy Q&A."*

---

## Phase 2 — LoRA fine-tuning + MLflow (1 day)  🎯 LLM fine-tuning, experiment tracking

Concept: instead of training a whole LLM (impossible on a laptop), **LoRA** trains a
tiny set of extra weights (adapters) — cheap, fast, runs on a free Colab GPU.

- Open `finetune/lora_finetune.ipynb` in **Google Colab** (free T4 GPU).
- It fine-tunes a small open model (e.g. `Phi-3-mini` or `Llama-3.2-3B`) with
  Hugging Face **PEFT + TRL** on an insurance Q&A dataset.
- Every run is logged to **MLflow** (loss, params, the LoRA adapter as an artifact).
- Download the adapter → we'll use it in Phase 3.

What you learn: Hugging Face Transformers/PEFT/TRL, LoRA, supervised fine-tuning (SFT),
MLflow tracking + model registry.

✅ *"Fine-tuned a small open LLM with LoRA (Hugging Face PEFT/TRL) and tracked all
experiments and the adapter registry in MLflow."*

---

## Phase 3 — Evaluation: RAGAS + LangSmith (½ day)  🎯 LLM/RAG evaluation

Concept: you can't ship an LLM you haven't measured. **RAGAS** scores your RAG on
faithfulness, answer relevancy, and context precision/recall. **LangSmith** traces every
call so you can debug the pipeline.

- Plug the fine-tuned adapter into the RAG (`src/rag.py`).
- Run `python eval/ragas_eval.py` on `data/qa_testset.jsonl` → get a metrics table.
- Turn on LangSmith tracing via `.env` and inspect traces in the dashboard.

What you learn: RAG evaluation metrics, RAGAS, LangSmith tracing/observability.

✅ *"Evaluated the RAG pipeline with RAGAS (faithfulness, context precision/recall) and
instrumented tracing with LangSmith."*

---

## Phase 4 — Kubernetes locally (½ day)  🎯 Kubernetes

Concept: Docker runs one container; **Kubernetes** runs, heals, and scales many.
Practice locally first with **minikube** (free, on your laptop).

- Build image: `docker build -t insureassist:latest .`
- `minikube start` → apply `k8s/` manifests → `kubectl get pods`
- Access the service, send a request.

What you learn: pods, Deployments, Services, ConfigMaps/Secrets, `kubectl`.

✅ *"Containerized the service and deployed it on Kubernetes (Deployment, Service,
ConfigMap/Secret) with health checks."*

---

## Phase 5 — Cloud on GCP (1 day)  🎯 Cloud (GCP), managed Kubernetes

- Store docs + LoRA adapter in **Cloud Storage (GCS)**.
- Push image to **Artifact Registry**.
- Create a **GKE** cluster, deploy the same `k8s/` manifests, expose a public URL.

What you learn: GCP fundamentals, GKE, GCS, Artifact Registry, cloud IAM basics.

✅ *"Deployed the system to Google Cloud (GKE + Cloud Storage + Artifact Registry) with a
public inference endpoint."*

---

## Phase 6 — CI/CD + autoscaling (½ day)  🎯 CI/CD, production scaling

- `.github/workflows/deploy.yml`: on push → build image → push → deploy to GKE.
- Add a **HorizontalPodAutoscaler** so pods scale with load.

✅ *"Automated build-and-deploy with GitHub Actions and configured Kubernetes HPA
autoscaling."*

---

## Phase 7 — Write-up (½ day)

- Polish `README.md`, add an architecture diagram + a short demo GIF/screenshots.
- Push to GitHub (github.com/mzquadri).
- Add ONE project entry to the CV (I'll help word it).

✅ Final CV line: *"InsureAssist — end-to-end fine-tuned RAG assistant deployed on GCP
(GKE) with LoRA fine-tuning, RAGAS evaluation, MLflow tracking, and CI/CD."*

---

### After this project, your CV honestly covers:
Cloud (GCP) · Kubernetes · LoRA/PEFT fine-tuning · Hugging Face · RAGAS/LangSmith eval ·
MLflow · CI/CD — **all the red/orange gaps, on top of skills you already have.**
