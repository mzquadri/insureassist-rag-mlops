# InsureAssist

![CI](https://github.com/mzquadri/insureassist-rag-mlops/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**A fine-tuned, cloud-native RAG assistant for insurance-policy questions — with the full
MLOps loop around it: fine-tuning, evaluation, containerization, Kubernetes, and CI/CD.**

Ask a question in plain English (*"Does my home policy cover a burst pipe?"*) and the
service retrieves the relevant policy clauses and answers with citations — grounded in the
documents, not made up.

> The sample data is generic insurance-policy Q&A, but the architecture is domain-agnostic:
> point it at legal, medical, or HR documents and it works the same way.

---

## Architecture

![System architecture](docs/architecture.svg)

At query time the flow is simple: embed the question, find the most similar policy chunks
in **Qdrant**, hand those chunks to the **LLM**, and return a grounded answer with its
sources. Documents are embedded and stored once, ahead of time.

---

## What's inside

- **Retrieval-Augmented Generation** — FastAPI service, Qdrant vector search, BGE embeddings.
- **A fine-tuned model** — a small open LLM (Phi-3-mini) adapted to the insurance domain
  with **LoRA (Hugging Face PEFT)**, trained on a free GPU and tracked with **MLflow**.
- **Real evaluation** — an **LLM-as-judge** harness that scores faithfulness, answer
  relevancy, context precision, and correctness (the same metrics RAGAS popularized).
- **Production packaging** — a **Docker** image and **Kubernetes** manifests (Deployment,
  Service, ConfigMap, HPA autoscaling), ready for **Google Kubernetes Engine (GKE)**.
- **Automation** — a **GitHub Actions** CI pipeline, plus a deploy workflow for the cloud.

---

## MLOps lifecycle

![MLOps lifecycle](docs/mlops-pipeline.svg)

---

## Quickstart (runs locally in ~10 minutes)

```bash
python -m venv .venv && .venv\Scripts\Activate.ps1     # Windows PowerShell
pip install -r requirements.txt
copy .env.example .env

docker compose up -d qdrant        # start the vector database (port 6533)
ollama pull llama3.2:3b            # local LLM for generation
python -m src.ingest              # documents -> embeddings -> Qdrant
uvicorn src.api:app --reload      # API at http://localhost:8000/docs
```

Then open http://localhost:8000/docs and try the `/ask` endpoint:

```text
Q: Does home insurance cover water damage from a burst pipe?
A: Yes — sudden and accidental water damage from a burst pipe is covered, including the
   cost of tracing and accessing the leak. Gradual leakage or wear and tear is not covered.
   sources: home_insurance_policy.md (0.87), home_insurance_policy.md (0.70), ...
```

Full, phase-by-phase instructions are in [`ROADMAP.md`](ROADMAP.md), and a plain-English
walkthrough of every file is in [`docs/PROJECT_EXPLAINED.md`](docs/PROJECT_EXPLAINED.md).

---

## Project structure

```
insurance-rag-mlops/
├── src/                    # RAG service
│   ├── api.py              #   FastAPI app (/ask, /health)
│   ├── rag.py              #   retrieve + generate
│   ├── ingest.py           #   documents -> embeddings -> Qdrant
│   ├── hf_generator.py     #   generation with the fine-tuned model
│   └── config.py           #   settings from .env
├── finetune/               # LoRA fine-tuning (Colab notebook) + guide
├── eval/                   # LLM-as-judge evaluation harness + sample report
├── k8s/                    # Kubernetes manifests + guide
├── data/                   # sample policies + Q&A test set
├── docs/                   # architecture diagrams, deep-dive, GCP deploy guide
├── .github/workflows/      # CI and deploy pipelines
├── Dockerfile              # container image
├── docker-compose.yml      # local Qdrant
└── requirements.txt
```

---

## Tech stack

| Area | Tools |
|---|---|
| ML / GenAI | PyTorch, Hugging Face Transformers, PEFT (LoRA), sentence-transformers (BGE) |
| RAG | Qdrant (vector DB), FastAPI |
| Fine-tuning & tracking | LoRA, MLflow |
| Evaluation | LLM-as-judge (faithfulness, relevancy, context precision, correctness) |
| Packaging & ops | Docker, Kubernetes, GitHub Actions |
| Cloud | Google Cloud — GKE, Cloud Storage, Artifact Registry |

---

## Project status

| Phase | Component | Status |
|---|---|---|
| 1 | RAG pipeline (FastAPI + Qdrant + BGE) | ✅ Working |
| 2 | LoRA fine-tuning (PEFT) + MLflow | ✅ Done |
| 3 | LLM-as-judge evaluation | ✅ Done |
| 4 | Docker image | ✅ Verified |
| 4 | Kubernetes manifests | ✅ Authored |
| 5 | GKE cloud deployment | ⏳ Guide in [`docs/gcp_deploy.md`](docs/gcp_deploy.md) |
| 6 | CI / CD (GitHub Actions) | ✅ CI green · deploy ready |

---

## License

Released under the [MIT License](LICENSE).
