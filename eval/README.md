# Phase 3 — RAG Evaluation (LLM-as-judge)

Measures the quality of the RAG pipeline. We score the same metrics the popular **RAGAS**
library implements, but with a **local LLM (Ollama) as the judge** — so it runs fully
offline, free, and with no fragile extra dependencies.

## Metrics
- **faithfulness** — is the answer supported by the retrieved context? (no hallucination)
- **answer_relevancy** — does the answer address the question?
- **context_precision** — are the retrieved chunks relevant to the question?
- **answer_correctness** — does the answer match the reference (ground-truth) answer?

Each is scored 0.0–1.0 by the judge; we report the average over the 10-question test set
(`data/qa_testset.jsonl`).

## Run
```bash
docker compose up -d qdrant     # vector DB (if not already running)
ollama pull llama3.2:3b         # judge + generator model
python -m src.ingest            # (only needed once)
python -m eval.evaluate
```
Results print to the console and are saved to `eval/eval_report.csv`.

## Example result (local llama3.2:3b judge)
| metric | score |
|---|---|
| faithfulness | 0.54 |
| answer_relevancy | 0.52 |
| context_precision | 0.80 |
| answer_correctness | 0.60 |

## Note on the judge
A small 3B model is a **noisy** judge — it hedges and is inconsistent on nuanced scores
(here, retrieval/`context_precision` is stable at 0.80, but faithfulness/relevancy vary).
A stronger judge (e.g. GPT-4-class via an API) gives more reliable, differentiated scores;
the judge is a single function (`_judge` in `evaluate.py`) and is easy to swap.

The takeaway is the **evaluation harness** itself: it turns "the answers look fine" into
measurable, repeatable numbers — which is exactly what RAG evaluation is for.
