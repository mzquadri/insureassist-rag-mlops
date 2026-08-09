# InsureAssist — Project Deep-Dive

A beginner-friendly, end-to-end walkthrough of how the project works and why each piece is
there. Read it top to bottom once, then use it as a reference. No prior knowledge of these
tools is assumed.

---

## 1. The one-sentence idea

> Ask an insurance question in plain English → the system finds the relevant policy text
> and an LLM writes a grounded, cited answer → all packaged, evaluated, and deployed to the
> cloud like a real product.

This pattern is called **RAG (Retrieval-Augmented Generation)**. It is the single most
common GenAI system in industry today.

---

## 2. Why RAG? (the problem it solves)

A plain LLM (like ChatGPT) has two problems for a company:
1. It doesn't know *your* private documents (your policies, contracts, manuals).
2. It can "hallucinate" — make up confident but wrong answers.

**RAG fixes both:** instead of trusting the LLM's memory, we *retrieve* the exact
paragraphs from your documents and tell the LLM: "answer using ONLY this text." The answer
is grounded in real sources and can be cited.

---

## 3. How it works at runtime (the data flow)

```
                    ┌──────────── INGESTION (done once, offline) ───────────┐
  policy docs  ──▶  split into chunks  ──▶  embed each chunk  ──▶  store in Qdrant
                                            (text → vector)        (vector database)
                    └───────────────────────────────────────────────────────┘

                    ┌──────────── QUERY (every question) ──────────────────┐
  question  ──▶ embed  ──▶ search Qdrant for nearest chunks ──▶ build prompt
                                                                    │
                                                            LLM generates answer
                                                                    │
                                                          answer + source list
                    └──────────────────────────────────────────────────────┘
```

Key terms:
- **Embedding**: a model turns text into a list of numbers (a *vector*) that captures its
  meaning. Similar meanings → nearby vectors. We use **BGE** (a small, strong embedder).
- **Vector database (Qdrant)**: stores those vectors and finds the closest ones fast. This
  is *semantic* search (by meaning), not keyword matching.
- **LLM**: the model that writes the final answer. Locally we use **Ollama**; in the cloud
  we use our **fine-tuned** model.

---

## 4. File-by-file (what each file does)

### `src/` — the application
| File | What it does |
|---|---|
| `config.py` | Reads settings from `.env` (URLs, model names, chunk sizes). One place for all knobs. |
| `ingest.py` | Loads docs from `data/`, splits into overlapping chunks, embeds them, stores in Qdrant. Run once to fill the database. |
| `rag.py` | The core: `retrieve()` (search Qdrant) + `build_prompt()` + `generate()` (call the LLM) = `answer()`. |
| `api.py` | Wraps `answer()` in a FastAPI web service with `/ask` and `/health` endpoints. |
| `hf_generator.py` | Phase 3: generation using the fine-tuned Hugging Face model (base + LoRA adapter). |

### `data/` — sample content
- `home_insurance_policy.md`, `auto_insurance_policy.md`: fake but realistic policies.
- `qa_testset.jsonl`: 10 questions with correct answers — used to *evaluate* the system.

### `finetune/` — teaching the LLM (Phase 2)
- `lora_finetune.ipynb`: a Colab notebook that fine-tunes a small LLM with **LoRA**.
- `README.md`: how to run it on Colab's free GPU.

### `eval/` — measuring quality (Phase 3)
- `ragas_eval.py`: runs the RAG on the test set and scores it with **RAGAS** metrics.

### `k8s/` — running it like a product (Phase 4/5)
- `qdrant.yaml`, `api-deployment.yaml`, `api-service.yaml`: run the DB + API on Kubernetes.
- `configmap.yaml` / `secret.example.yaml`: configuration and secrets.
- `hpa.yaml`: autoscaling (add pods when busy).

### root & ops
| File | What it does |
|---|---|
| `docker-compose.yml` | Starts Qdrant locally (isolated: port 6533, name `insureassist-qdrant`). |
| `Dockerfile` | Packages the API into a container image. |
| `docs/gcp_deploy.md` | Step-by-step cloud deploy on Google Cloud (GKE). |
| `.github/workflows/ci.yml` | Runs on every push: syntax + lint + YAML checks (green ✅). |
| `.github/workflows/deploy.yml` | Manual-only: builds image and deploys to GKE (needs GCP secrets). |

---

## 5. The phases (each adds one capability)

| Phase | You learn | Runs where |
|---|---|---|
| 1. RAG baseline | embeddings, vector DB, FastAPI | your laptop |
| 2. LoRA fine-tuning | Hugging Face PEFT/TRL, MLflow | Google Colab (free GPU) |
| 3. Evaluation | RAGAS, LangSmith tracing | your laptop |
| 4. Kubernetes | Docker → k8s, minikube | your laptop |
| 5. Cloud | GCP: GKE, Cloud Storage, Artifact Registry | Google Cloud (free credit) |
| 6. CI/CD | GitHub Actions, autoscaling | GitHub |

Follow `ROADMAP.md` for the exact commands of each phase.

---

## 6. Glossary (quick reference)

- **RAG**: retrieve relevant text, then let the LLM answer using it (grounded, cited).
- **Embedding**: text → vector capturing meaning.
- **Vector database**: stores vectors, finds nearest neighbours (semantic search).
- **Chunking**: splitting documents into small pieces so retrieval is precise.
- **LLM fine-tuning**: adapting a model to your domain/style.
- **LoRA / PEFT**: fine-tune only tiny extra weights — cheap, fast, small file.
- **Quantization (4-bit)**: store model weights compactly to fit a small GPU.
- **RAGAS**: automatic RAG quality metrics (faithfulness, relevancy, context precision/recall).
- **MLflow**: tracks experiments (params, metrics, model artifacts).
- **Docker**: package an app + its dependencies into a portable container.
- **Kubernetes**: runs, heals, and scales many containers.
- **GKE**: Google's managed Kubernetes.
- **CI/CD**: automatically test (CI) and deploy (CD) on every code change.
- **HPA**: Kubernetes autoscaler that adds/removes pods based on load.

---

## 7. Skills demonstrated

Working through the project touches the core of a modern applied-AI stack:

- **Retrieval-Augmented Generation** — embeddings, vector search, grounded generation.
- **LLM fine-tuning** — LoRA / PEFT, and experiment tracking with MLflow.
- **Evaluation** — measuring RAG quality with an LLM-as-judge instead of eyeballing it.
- **MLOps & deployment** — Docker, Kubernetes, Google Cloud (GKE), and CI/CD.

Each phase is small enough to understand end to end, which is the point: not just wiring
libraries together, but knowing *why* each piece is there.
