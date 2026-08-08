# InsureAssist — Fine-Tuned, Cloud-Native RAG Assistant with Evaluation & MLOps

An end-to-end **Retrieval-Augmented Generation (RAG)** assistant for insurance/policy
question-answering, built to production standards. It extends common RAG work with the
skills that AI/ML Engineer job postings ask for most: **cloud deployment, Kubernetes,
LLM fine-tuning (LoRA/PEFT), RAG evaluation, and experiment tracking.**

> Domain note: the sample data here is generic insurance-policy Q&A so it maps directly
> to the kind of document-intelligence work done in industry, but you can swap in any
> corpus (legal, medical, HR) without changing the architecture.

---

## What this project demonstrates (maps 1:1 to CV gaps)

| Gap it fills | Where in this project |
|---|---|
| **Cloud (GCP)** | GKE (Kubernetes), Cloud Storage (model + docs), Artifact Registry (images) |
| **Kubernetes** | `k8s/` manifests: Deployment, Service, ConfigMap, Secret, HPA autoscaling |
| **LLM fine-tuning (LoRA/PEFT)** | `finetune/` — Hugging Face PEFT + TRL SFT on a small open model |
| **RAG evaluation (RAGAS / LangSmith)** | `eval/` — RAGAS metrics + LangSmith tracing |
| **Experiment tracking (MLflow)** | fine-tuning runs, eval metrics, and the LoRA adapter registered in MLflow |
| Reinforces | Hugging Face Transformers, Docker, FastAPI, Qdrant vector DB, CI/CD (GitHub Actions) |

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │                 GKE (Kubernetes)             │
  User ──HTTP──▶  FastAPI  ──retrieve──▶  Qdrant (vector DB)       │
                    │  │                                           │
                    │  └──generate──▶  Fine-tuned LLM (LoRA)       │
                    └───────┬─────────────────────┬───────────────┘
                            │                     │
                     GCS bucket            MLflow (tracking +
                 (docs + LoRA adapter)      model registry)

  Offline: RAGAS evaluation + LangSmith tracing on a test question set
  CI/CD:   GitHub Actions → build image → push to Artifact Registry → deploy to GKE
```

---

## Tech stack

- **LLM / GenAI:** Hugging Face Transformers, PEFT (LoRA), TRL, sentence-transformers (BGE)
- **RAG:** Qdrant (vector DB), FastAPI serving
- **Evaluation:** RAGAS, LangSmith
- **MLOps:** MLflow (tracking + registry), Docker, Kubernetes, GitHub Actions
- **Cloud:** Google Cloud (GKE, Cloud Storage, Artifact Registry)

---

## Repo layout

```
insurance-rag-mlops/
├── src/           # RAG app: ingestion, retrieval+generation, FastAPI
├── data/          # sample insurance policy docs + Q&A test set
├── finetune/      # LoRA fine-tuning (runs on Google Colab free GPU)
├── eval/          # RAGAS + LangSmith evaluation
├── k8s/           # Kubernetes manifests
├── .github/workflows/  # CI/CD pipeline
├── requirements.txt
├── docker-compose.yml
└── ROADMAP.md     # step-by-step learning plan (START HERE)
```

**New here? Open `ROADMAP.md` and start at Phase 0.**
