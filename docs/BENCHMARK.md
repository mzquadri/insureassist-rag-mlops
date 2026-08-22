# NFIP retrieval benchmark

The first benchmark in this repository that measures anything. It asks one question: **given
a question about a flood policy, does the retriever return the passage that actually answers
it?**

Everything here is dense retrieval only — the existing BGE retriever, unchanged. There is no
lexical baseline, no hybrid, no reranking, and no generation scoring yet. This is the
before picture.

```bash
docker compose up -d qdrant
QDRANT_COLLECTION=nfip_sfip python -m src.ingest
python -m eval.validate_ground_truth
QDRANT_COLLECTION=nfip_sfip python -m eval.reference_run --split test
python -m eval.failure_analysis
```

## What is measured against

- **Corpus:** 3 NFIP policy forms, 35,639 words, **426 chunks** (see [`DATA.md`](DATA.md)).
- **Questions:** 40 hand-authored labels, 32 answerable and 8 unanswerable (20%).
- **Split:** deterministic, hashed from the question ID and stratified by category — dev 18,
  test 22. Dev is for tuning decisions; **test is reported and not tuned on**.
- **Labels:** every question names the exact chunk IDs and character offsets holding its
  evidence. `python -m eval.validate_ground_truth` fails if any reference stops resolving.

## Results — test split, dense retrieval

| Metric | @1 | @3 | @5 |
|---|---|---|---|
| Hit rate | 0.167 | 0.556 | 0.611 |
| Recall | 0.167 | 0.528 | 0.556 |
| Precision | 0.167 | 0.185 | 0.122 |

**MRR 0.366** · **top-document accuracy 0.167** · latency p50 41 ms, p95 48 ms.

Dev split, for comparison: hit rate@5 0.643, MRR 0.413, top-document accuracy 0.357.

Precision is bounded by construction: most questions have one relevant chunk, so
precision@5 cannot exceed 0.2. It is reported for completeness, not as a headline.

### The finding

**Top-document accuracy is 0.167.** For five of every six questions, the highest-scoring
chunk comes from *the wrong policy form*.

That is the benchmark doing its job. The three forms are near-duplicates by design — they
share a skeleton and much verbatim wording — so a retriever matching on topic alone will
find the $30,000 Increased Cost of Compliance limit, or the $2,500 special limit, or the
60-day proof-of-loss deadline, in whichever form happens to embed closest. All three
contain that provision. Only one of them is the form the question asked about.

Failure analysis over the 18 answerable test questions:

| Failure class | Count | Share |
|---|---|---|
| Wrong near-duplicate form | 7 | 39% |
| Multi-chunk partial miss | 2 | 11% |
| **Total failures** | **9** | **50%** |

Not one failure was "irrelevant retrieval". The retriever consistently finds the right
*provision* and the wrong *document*.

### By question category (test split)

| Category | n | Hit rate@5 | MRR |
|---|---|---|---|
| time_period | 4 | 1.00 | 0.583 |
| multi_chunk | 2 | 1.00 | 0.375 |
| exclusion | 3 | 0.667 | 0.333 |
| numeric_limit | 2 | 0.50 | 0.500 |
| single_chunk | 2 | 0.50 | 0.500 |
| **near_miss** | **5** | **0.20** | **0.100** |

The near-miss questions — the ones deliberately built so that all three forms look
plausible — score 0.20. Everything else scores 0.5 to 1.0. The gap is the whole result.

### By form

| Form | n | Hit rate@5 | MRR |
|---|---|---|---|
| Dwelling | 8 | 0.750 | 0.500 |
| General Property | 7 | 0.571 | 0.226 |
| RCBAP | 3 | 0.333 | 0.333 |

RCBAP has only 3 answerable test questions, so its figure is indicative at best.

## Unanswerable questions

Reported separately and **never folded into recall**. A question with no relevant chunk has
no recall to measure; scoring it zero would invent a failure and scoring it one would invent
a success.

| | Value |
|---|---|
| Count (test) | 4 |
| Top-1 similarity, mean | 0.719 |
| Top-1 similarity, range | 0.658 – 0.776 |
| Answerable top-1 similarity, mean | 0.775 |

**No abstention threshold is proposed, and none should be inferred from these numbers.**
The distributions overlap heavily, and the worst case is instructive: `nfip-033` asks for
the coinsurance threshold *under the Dwelling Form*, which has no coinsurance provision at
all. Its top-1 similarity is **0.776 — above the mean for answerable questions**. Any naive
similarity cut-off would confidently pass that question straight through to a generator.

Setting a threshold requires a study on the dev split. That has not been run.

## What these numbers do not say

- Nothing about **answer quality**. This measures retrieval only.
- Nothing about **citation correctness** — the API still returns document-level attribution.
- Nothing about **other insurance domains**. One jurisdiction, one peril, three documents.
- Nothing about **whether dense is the right choice** — there is no baseline to compare
  against yet. A lexical baseline is the obvious next step, because the observed failure is
  exactly the kind that exact-term matching may handle differently.

## Reproducing

`eval/reference_run_nfip.json` records the model, dimension, distance, chunk size and
overlap, top-k, corpus hash, config hash and question-set hash alongside every per-question
result. The corpus and labels are committed and hash-verified, so the run can be repeated
and compared. Latency is machine-dependent; everything else is not.
