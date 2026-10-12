# RAG evaluation (local LLM-as-judge)

A harness that scores generated answers with a local model through Ollama, so it runs
offline and needs no API key.

> **This documents the original harness, on a different corpus from the rest of the
> repository.** It scores 10 generic questions (`data/qa_testset.jsonl`) against the two
> synthetic policies in `data/auto_insurance_policy.md` and `data/home_insurance_policy.md`.
> The evaluation the project's claims actually rest on uses the three real NFIP flood forms
> in `data/corpus/`, 50 labelled questions, and 314 chunks — see
> [`docs/BENCHMARK.md`](../docs/BENCHMARK.md), [`docs/LIMITATIONS.md`](../docs/LIMITATIONS.md)
> and the arms indexed at the bottom of this file. Nothing below describes that work, and
> the two should not be read together.

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

**This harness measures no retrieval quality.** `data/qa_testset.jsonl` contains reference
*answers* only, with no relevant-document or relevant-chunk labels, so Recall@k, Hit
Rate@k, MRR and nDCG cannot be computed from it, and there is no lexical baseline to
compare against.

> An earlier version of this sentence read "Retrieval quality is not measured anywhere in
> this repository." That was true when written and is now false. The NFIP ground truth in
> `eval/ground_truth/nfip_questions.jsonl` carries `relevant_chunk_ids` and
> `evidence_spans` for all 50 questions, and `eval/reference_run.json` reports hit@5, MRR
> and top-document accuracy against BM25 and dense-only baselines. The claim is corrected
> rather than deleted, because a reader who saw the old wording should be able to find out
> that it changed.

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

---

## The NFIP evaluation arms

What the project's published claims rest on. Each writes a pinned artefact, and
`python -m eval.verify_artifacts` fails if a documented number drifts from it.

| Arm | Code | Artefact | Judge? |
|---|---|---|---|
| Retrieval benchmark | [`eval/reference_run.py`](reference_run.py) | `reference_run.json` | no |
| Abstention threshold (tested, rejected) | [`eval/abstention_threshold.py`](abstention_threshold.py) | `abstention_threshold.json` | no |
| Citation structure — does it cite blocks it was given | [`eval/citation_structure.py`](citation_structure.py) | `citation_structure.json` | no |
| Prompt variant, same pinned retrieval | [`eval/prompt_variants.py`](prompt_variants.py) | `citation_structure_ranged.json` | no |
| Citation placement — where the marker sits | [`eval/citation_placement.py`](citation_placement.py) | `citation_placement.json` | no |
| Citation correctness — ALCE recall/precision | [`eval/citation_correctness.py`](citation_correctness.py) | `citation_correctness.json` | NLI, calibrated |
| Judge-panel spread — how far the result depends on the instrument | [`eval/judge_panel.py`](judge_panel.py) | `judge_panel.json` | no (reads both runs) |
| Human gold standard for the above | [`eval/citation_gold.py`](citation_gold.py) | `citation_gold.json` | a person |

Six of the eight need no judge at all, which is the point: they are facts about strings
and labels, so the circularity objection that rules out self-grading does not reach them.
The one arm that does use a judge calibrates it against the ground truth first and refuses
to report a rate if it misses floors declared before the run.

`citation_gold.json` does not exist yet. Its worksheet does
(`eval/citation_gold/worksheet.md`, built by `python -m eval.citation_gold_run --build`)
and is unlabelled; the judge has never been checked against a person on the pairs it
scored, which is the sharpest open caveat in `docs/LIMITATIONS.md`.

The worksheet uses AttrScore's three labels — attributable, extrapolatory, contradictory
(Yue et al., EMNLP Findings 2023, arXiv:2305.06311) — rather than a binary, because
missing evidence and contradicting evidence are different failures pointing at different
parts of the system. Only `attributable` counts as support, so the ALCE rates are
unaffected. The five sentences the two judges disagree about are ordered first: they are
the only ones that can settle which judge to believe, so a pass that stops early still
answers the open question.
