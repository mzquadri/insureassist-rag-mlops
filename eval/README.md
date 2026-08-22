# RAG evaluation (local LLM-as-judge)

A harness that scores generated answers with a local model through Ollama, so it runs
offline and needs no API key.

> **These are custom judge metrics, not RAGAS metrics.**
>
> Earlier wording here described them as "the same metrics the RAGAS library implements".
> That was wrong and has been removed. RAGAS computes faithfulness by decomposing an answer
> into atomic claims and verifying each one; it computes context precision as a
> rank-weighted average over individual chunks; it computes answer correctness from
> statement-level true/false positives. **None of that happens here.** Each metric below is
> a single holistic 0-1 score from one prompt. The names are borrowed; the algorithms are
> not. `ragas` is not a dependency of this project.

## What is measured

| Metric | Question it asks |
|---|---|
| `judge_faithfulness` | Is the answer supported by the retrieved context? |
| `judge_answer_relevancy` | Does the answer address the question? |
| `judge_answer_correctness` | Does the answer match the reference answer? |

## What is not measured

**Retrieval quality is not measured anywhere in this repository.**

`data/qa_testset.jsonl` contains reference *answers* only. There are no relevant-document
or relevant-chunk labels, so Recall@k, Hit Rate@k, MRR and nDCG cannot be computed. There
is no lexical baseline to compare against either.

`context_precision` used to be reported here as a retrieval signal. **It has been withdrawn
from the headline table.** It scored exactly `0.80` on all ten questions, which is not a
retriever performing consistently - it is a metric carrying no information. Two things
caused it: the judge sees all retrieved chunks concatenated into one blob, so ranking is
invisible to it; and with only 7 chunks in the whole corpus and `TOP_K=4`, more than half
the index is returned for every query, so the context is nearly always partly relevant.

The per-question value stays in `eval_report.csv` for transparency. It should not be read
as evidence about retrieval.

## Known weaknesses of this harness

- **The judge is the model being judged.** `JUDGE_MODEL = cfg.OLLAMA_MODEL`, so the same
  model both writes and grades the answer. Self-assessment is a known source of bias.
- **A 3B judge is coarse.** In practice it emits only `{0.0, 0.5, 0.8, 0.9}` - roughly a
  four-level scale, not a continuous score.
- **Answers are not deterministic.** Generation runs at Ollama's default temperature, so
  scores move between runs. The judge itself runs at `temperature: 0`.
- **Single run, no variance.** No repeats, no confidence intervals, no seed.
- **Unparsable judge output becomes NaN and is dropped from the average**, and the printed
  summary does not say how many were dropped - so an average can rest on fewer than ten
  questions without saying so.
- **Retrieval and generation failures are not separable.** A wrong answer could mean the
  retriever missed the evidence or the generator ignored it, and nothing here tells them
  apart. Question 8 in the report is a live example: the model answered *"Don't know...
  does not mention mechanical breakdown"* when the auto policy does exclude mechanical
  breakdown in its Exclusions section. The evidence existed and was not retrieved, yet
  `context_precision` still scored 0.80 for that row.

## Run

```bash
docker compose up -d qdrant     # vector DB
ollama pull llama3.2:3b         # generator and judge
python -m src.ingest            # only needed once
python -m eval.evaluate
```

Results print to the console and are written to `eval/eval_report.csv`.

## Recorded result (local llama3.2:3b judge, single run, 10 questions)

| metric | score |
|---|---|
| judge_faithfulness | 0.54 |
| judge_answer_relevancy | 0.52 |
| judge_answer_correctness | 0.60 |

These averages are computed from the committed `eval_report.csv` and can be recomputed
from it. They describe **the stock `llama3.2:3b` model**, not a fine-tuned one, and they
say nothing about retrieval quality. Given a self-judging 3B model on ten questions from a
7-chunk corpus, treat them as a smoke test of the harness rather than a measurement of the
system.
