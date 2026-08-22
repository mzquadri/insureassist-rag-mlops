# NFIP retrieval benchmark

One question: **given a question about a flood policy, does the retriever return the passage
that answers it?**

Every number here is derived from [`eval/reference_run.json`](../eval/reference_run.json).
`python -m eval.verify_artifacts` fails if this document drifts from it, and CI runs that.

## What it measures against

- **Corpus:** 3 NFIP policy forms, 35,639 words, **314 chunks** at 800/120 ([`DATA.md`](DATA.md)).
- **Questions:** 40 hand-authored labels — 32 answerable, 8 unanswerable (20%).
- **Split:** deterministic hash of the question ID, stratified by category. Dev 18, test 22.
- Dev drove every architecture and configuration decision. **Test was run once, after the
  configuration was frozen in [`eval/retrieval_config.json`](../eval/retrieval_config.json).**

## Final architecture

**Hybrid: dense BGE + Okapi BM25, fused by reciprocal rank fusion.** Chunking 800/120,
20 candidates per retriever, RRF k=60, serving top_k=5.

## Held-out TEST results

| Metric | @1 | @3 | @5 |
|---|---|---|---|
| Hit rate | 0.278 | 0.500 | **0.556** |
| Recall | 0.241 | 0.407 | 0.463 |
| Precision | 0.278 | 0.167 | 0.111 |

MRR **0.420** · top-document accuracy **0.556**

### Baseline comparison, same corpus, same labels, same split

| | Hit@5 | MRR | Top-doc |
|---|---|---|---|
| Dense only | 0.500 | 0.365 | 0.556 |
| BM25 only | **0.611** | 0.366 | 0.333 |
| **Hybrid RRF (selected)** | 0.556 | **0.420** | **0.556** |
| *Dense at the original 600/100 config* | *0.611* | *0.366* | *0.167* |

**Read this honestly.** The selected architecture wins on MRR and ties the best
top-document accuracy, but **BM25 alone retrieves more relevant chunks in the top 5**. The
headline improvement over the starting point is top-document accuracy: **0.167 → 0.556**,
a 3.3× reduction in returning the wrong policy form. Overall hit@5 did not improve on the
original dense baseline.

### Dev did not generalise

Hybrid scored hit@5 **1.000** on dev and **0.556** on test. That gap is the honest cost of
selecting on 14 answerable questions. The dev figure was flagged as small-sample when the
configuration was frozen, and it is reported here rather than quietly dropped.

### By category (test)

| Category | n | Hit@5 | MRR |
|---|---|---|---|
| time_period | 4 | 1.000 | 1.000 |
| multi_chunk | 2 | 1.000 | 0.750 |
| exclusion | 3 | 0.667 | 0.333 |
| near_miss | 5 | 0.400 | 0.150 |
| numeric_limit | 2 | 0.000 | 0.083 |
| single_chunk | 2 | 0.000 | 0.071 |

Near-miss questions remain the weakest substantial category — the three forms are still
being confused, just less often. `numeric_limit` and `single_chunk` are n=2 each and should
not be read as findings.

Per-form numbers are in the artefact and are **not reproduced here**: with 3–8 answerable
questions per form they carry no useful signal.

## Citations

Deterministic, no model involved.

| | Value |
|---|---|
| Citation precision | 0.111 |
| Citation recall | 0.463 |
| **Unsupported citation rate** | **0.000** |
| Citations checked | 90 |

Precision is bounded by construction: most questions have one relevant chunk and five are
returned, so 0.2 is the ceiling. The number that matters is **unsupported = 0.000** — every
citation's offsets reproduce its quoted text from the committed corpus. No provenance is
fabricated.

## Abstention

| | Value |
|---|---|
| Answerable acceptance | 1.000 |
| Unanswerable rejection | **0.000** |
| False abstentions | 0 |
| False answers | 4 |

**The service does not detect unanswerable questions, and no threshold is claimed.** On dev,
the best single-threshold rule on top-1 dense similarity scored 0.778 — exactly the
always-answer baseline. The hardest trap (`nfip-034`: coinsurance asked of a form that has
no coinsurance article) scored 0.787, *above* the answerable mean of 0.761. Any cut-off
catching it would reject most real questions.

Only the structural case is implemented: retrieval returns nothing → `insufficient_evidence`.
On this corpus that never fires, so unanswerable rejection is 0.000. This is a real
weakness, recorded as one.

## What these numbers do not say

- Nothing about **answer quality**. No judge metrics are published — only one local model is
  available and grading its own output is circular.
- Nothing about **other domains**. One jurisdiction, one peril, three documents.
- **Latency** (p50 ~50 ms local) is machine-specific and is not a system property.

## Reproducing

```bash
docker compose up -d qdrant
QDRANT_COLLECTION=nfip_sfip python -m src.ingest
python -m eval.validate_ground_truth
QDRANT_COLLECTION=nfip_sfip python -m eval.reference_run --split test
python -m eval.failure_analysis --run eval/reference_run.json
python -m eval.verify_artifacts
```

Dev-selection evidence is in [`eval/dev_comparison.json`](../eval/dev_comparison.json) and
[`eval/dev_chunking_sweep.json`](../eval/dev_chunking_sweep.json). The pre-selection dense
baseline is preserved under [`eval/baselines/`](../eval/baselines/).
