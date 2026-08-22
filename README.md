# InsureAssist

![CI](https://github.com/mzquadri/insureassist-rag-mlops/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**A retrieval-augmented question-answering service for insurance policy documents, built as
an end-to-end engineering exercise: ingestion, vector search, a served API, a container, an
evaluation harness, and Kubernetes manifests.**

Ask a question in plain English (*"Does my home policy cover a burst pipe?"*) and the
service retrieves the most similar policy passages and answers from them, returning the
documents it drew on.

> **Read this before the feature list.** Retrieval is now measured against a real corpus:
> the three NFIP Standard Flood Insurance Policy forms published as federal regulation
> (35,639 words, 426 chunks), with 40 hand-labelled questions. The headline result is not
> flattering — the retriever picks the **correct policy form only 17% of the time**, because
> the three forms are near-duplicates. See [`docs/BENCHMARK.md`](docs/BENCHMARK.md).
>
> Generation quality, citation correctness and abstention are still **not** measured.

---

## Architecture

![System architecture](docs/architecture.svg)

Embed the question, find the most similar policy chunks in **Qdrant**, hand those chunks to
a local **LLM**, return the answer with its source documents. Documents are embedded and
stored ahead of time by a separate ingestion step.

---

## What's inside

- **Retrieval-augmented generation** — FastAPI service, Qdrant vector search, BGE
  embeddings, generation through a local Ollama model (`llama3.2:3b`).
- **An answer-quality harness** — a local **LLM-as-judge** that scores faithfulness,
  answer relevancy and correctness. These are custom judge metrics, *not* RAGAS metrics,
  and they do not measure retrieval. See [`eval/README.md`](eval/README.md).
- **A LoRA fine-tuning notebook** — a Colab recipe adapting Phi-3-mini with PEFT and
  logging to MLflow. **Authored, not evidenced**: no adapter, no run data, and no evidence
  it completed. The served system does not use a fine-tuned model.
- **Container and orchestration assets** — a **Dockerfile**, and **Kubernetes** manifests
  (Deployment, Service, ConfigMap, HPA) plus a **GKE** deployment guide. The manifests and
  the guide are **authored, not deployed** — neither has been verified against a cluster.
- **CI** — a GitHub Actions workflow running **lint and static validation only**.

---

## Honest status

The table below is the authoritative statement of what exists. Where something is authored
but unproven, it says so.

| Component | State | Evidence |
|---|---|---|
| RAG pipeline (FastAPI + Qdrant + BGE + Ollama) | **Working** | `src/`, runs locally |
| Offline unit + API failure-path tests | **Working** | `tests/`, run in CI |
| Answer-quality judge harness | **Working, weak** | `eval/eval_report.csv`; self-judging 3B model, 10 questions, single run |
| Retrieval quality (NFIP) | **Measured** | 40 labelled questions, 426-chunk real corpus; hit rate@5 0.611, MRR 0.366, top-document accuracy 0.167 |
| Lexical / hybrid baseline | **Not built** | Dense retrieval has nothing to be compared against yet |
| Citation correctness | **Not measured** | `sources` is document-level; no chunk-level citation yet |
| Abstention | **Not measured** | Similarity distributions overlap; no threshold study run |
| LoRA fine-tuning + MLflow | **Authored, not evidenced** | Notebook has no outputs; no adapter; no run data |
| Docker image | **Builds and runs locally** | Manual verification only — not built or tested in CI |
| Kubernetes manifests | **Authored, not deployed** | YAML parses in CI; never applied to a cluster |
| GKE deployment | **Guide only** | [`docs/gcp_deploy.md`](docs/gcp_deploy.md); deploy workflow has never succeeded |
| CI | **Lint and static checks + offline tests** | [`ci.yml`](.github/workflows/ci.yml) |
| Deploy workflow | **Unproven** | Requires unset GCP secrets; both historical runs failed |

### Known limitations

- **Retrieval is measured, and it is weak.** On the held-out test split the retriever
  returns the right passage in the top 5 for 61% of answerable questions, and its single
  best hit comes from the wrong policy form 83% of the time.
- **There is no baseline.** Dense retrieval has not been compared with lexical or hybrid
  retrieval, so "dense is the right choice here" is unproven.
- **nDCG is not reported.** The labels are binary, so it would add a column rather than
  information.
- **The synthetic sample corpus (7 chunks) is retained only as an offline fixture.** No
  metric is published from it.
- **The evaluation judge is the model being judged**, which biases the scores.
- **The fine-tuning data is the evaluation set.** The ten training pairs in the notebook
  are the ten test questions. No fine-tuned model can be honestly scored against them —
  see [`finetune/README.md`](finetune/README.md).
- **`sources` are document names, not citations.** There is no chunk-level or span-level
  attribution yet, so citation correctness cannot be checked.
- **The service cannot abstain reliably.** The prompt asks the model to say when it does
  not know, but nothing enforces it and there is no evidence threshold.
- **`/health` is liveness only.** It reports nothing about Qdrant or the generator, so a
  pod can pass it and still be unable to answer. The Kubernetes manifests currently use it
  for readiness too, which is wrong and tracked.
- **The container runs as root**, has no `HEALTHCHECK`, and installs unpinned dependencies,
  so images are not reproducible.

---

## Quickstart (about 10 minutes)

```bash
python -m venv .venv && .venv\Scripts\Activate.ps1     # Windows PowerShell
pip install -r requirements.txt
copy .env.example .env

docker compose up -d qdrant        # vector database on host port 6533
ollama pull llama3.2:3b            # local LLM for generation
python -m src.ingest               # documents -> embeddings -> Qdrant
uvicorn src.api:app --reload       # API at http://localhost:8000/docs
```

Then open http://localhost:8000/docs and try `/ask`:

```text
Q: Does home insurance cover water damage from a burst pipe?
A: Yes - sudden and accidental water damage from a burst pipe is covered, including the
   cost of tracing and accessing the leak. Gradual leakage or wear and tear is not covered.
   sources: home_insurance_policy.md (0.87), home_insurance_policy.md (0.70), ...
```

Answers vary between runs: generation uses the Ollama default temperature.

Phase-by-phase build notes are in [`ROADMAP.md`](ROADMAP.md); a walkthrough of every file is
in [`docs/PROJECT_EXPLAINED.md`](docs/PROJECT_EXPLAINED.md).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Every test is offline. Nothing downloads a model, contacts Qdrant, or needs Ollama —
`import src.api` performs no model loading or network access, which is what makes that
possible. See [`tests/README.md`](tests/README.md).

---

## Project structure

```
insureassist-rag-mlops/
├── src/                    # RAG service
│   ├── api.py              #   FastAPI app (/ask, /health)
│   ├── rag.py              #   retrieve + generate
│   ├── ingest.py           #   documents -> embeddings -> Qdrant
│   ├── providers.py        #   lazy, injectable embedder / vector store / generator
│   ├── errors.py           #   dependency failures -> 503
│   ├── hf_generator.py     #   optional Hugging Face backend (needs its own deps)
│   └── config.py           #   settings from .env
├── tests/                  # offline unit + API tests
├── finetune/               # LoRA notebook (authored, not evidenced)
├── eval/                   # LLM-as-judge harness + recorded report
├── k8s/                    # Kubernetes manifests (authored, not deployed)
├── data/
│   ├── corpus/             #   real NFIP policy forms (CFR) + manifest
│   └── *.md                #   synthetic sample docs (fixtures only)
├── scripts/                # corpus fetch + ground-truth builder
├── docs/                   # diagrams, walkthrough, GCP guide
├── .github/workflows/      # CI and deploy pipelines
├── Dockerfile
├── docker-compose.yml      # local Qdrant
└── requirements.txt
```

---

## Tech stack

| Area | Tools |
|---|---|
| RAG | Qdrant (vector DB), sentence-transformers (BGE), FastAPI |
| Generation | Ollama (`llama3.2:3b`) |
| Retrieval evaluation | Labelled ground truth, Recall/Hit-rate/MRR/Precision@k |
| Answer evaluation | Custom LLM-as-judge metrics |
| Fine-tuning (notebook only) | Hugging Face Transformers, PEFT (LoRA), MLflow |
| Packaging & ops | Docker, Kubernetes manifests, GitHub Actions |
| Cloud (guide only) | Google Cloud — GKE, Artifact Registry |

`torch`, `transformers` and `peft` are used only by the optional `hf` backend and the Colab
notebook. They are not installed by `requirements.txt`.

---

## Benchmark

```bash
docker compose up -d qdrant
QDRANT_COLLECTION=nfip_sfip python -m src.ingest
python -m eval.validate_ground_truth
QDRANT_COLLECTION=nfip_sfip python -m eval.reference_run --split test
python -m eval.failure_analysis
```

Results, method and limitations: [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

## License

The [MIT License](LICENSE) covers **the code only**. The NFIP policy forms under
`data/corpus/` are US Government works carrying no copyright (17 U.S.C. 105) and are not
relicensed by this repository — see [`NOTICE`](NOTICE) and [`docs/DATA.md`](docs/DATA.md).
The synthetic sample data is self-authored; see [`data/README.md`](data/README.md).
