# Limitations

The complete list. Nothing here is softened.

## Benchmark

- **40 questions, 22 in the test split.** Small. Category cells of n=2-5 carry no reliable
  signal, and per-form numbers (3-8 questions each) are not reported as findings.
- **Dev did not generalise**: hit@5 1.000 on dev, 0.556 on test.
- **One jurisdiction, one peril, three documents.** Results describe US flood policy wording,
  not insurance documents generally.
- **Labels are binary**, so no graded relevance and no nDCG.

## Retrieval

- **BM25 alone beats the selected hybrid on hit@5** (0.611 against 0.556). Hybrid was chosen
  for MRR (0.420 against 0.366) and top-document accuracy (0.556 against 0.333). The
  trade-off is real and is not presented as a clean win.
- **hit@5 did not improve** on the original dense baseline at the old chunking (0.611). What
  improved is form discrimination: top-document accuracy 0.167 to 0.556.
- **Near-miss questions remain the weakest substantial category** at 0.400 hit@5. The three
  forms are still being confused, just less often.
- Chunking was selected from five configurations on dev; a wider sweep might do better.

## Abstention

- **Unanswerable rejection rate is 0.000.** The service answers every unanswerable question.
- No threshold is defensible on this data, and none is claimed.
- 8 unanswerable questions in total is too few to support any threshold claim.

## Generation

- **No answer-quality metric is published.** Only one local model is available, and grading
  its own output is circular.
- Generation is non-deterministic (default temperature) and is not part of the reference gate.
- The model is prompted to cite context numbers, but **whether it does so correctly is not
  measured**. The citation metrics score *retrieval*, not the generator's use of it.

## Ingestion

- **Trailing duplicate chunk.** A document whose length lands just past a chunk boundary
  emits a final chunk wholly contained in the previous one. Retained deliberately: fixing it
  changes every chunk ID, which invalidates all 40 labels and the reference run. The benefit
  does not justify the migration. It is pinned by a test so it cannot change silently.
- Chunking is character-based, not token-based or structure-aware.

## Operations

- Kubernetes manifests have never been applied to a cluster.
- GKE guidance is untested; the deploy workflow has never succeeded.
- In-cluster Ollama model provisioning is manual.
- The embedding model downloads on first use, so a cold container needs network access.
- No horizontal scaling evidence.

## History, kept on purpose

- The original 10-question test set **was** the fine-tuning training data. Fine-tuning is
  archived for that reason, and no tuned-model number exists anywhere.
- Earlier documentation claimed RAGAS metrics, a multi-stage Dockerfile, GKE readiness, CI
  that tested the pipeline, and a fine-tuned served model. None was true. All were corrected,
  and the corrections are recorded rather than quietly applied.
