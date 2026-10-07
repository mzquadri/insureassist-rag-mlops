# Limitations

The complete list. Nothing here is softened.

## Benchmark

- **50 questions, 27 in the test split, 18 of them answerable.** Small. Category cells of n=2-5 carry no reliable
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
- **A threshold was tested properly and rejected.** The previous entry said eight unanswerable
  questions were too few to support any claim, which was true and was the wrong place to
  stop. The set now holds 18, each one verified silent against the corpus by
  `eval/check_unanswerable.py` rather than assumed silent. A top-1 dense threshold was chosen
  on dev by Youden's J and applied unchanged to test.
- **It does not survive the move to held-out data**: balanced accuracy 0.802 on dev, 0.556 on
  test. The dev number was memorisation.
- **The two populations overlap almost entirely.** 17 of the 18 unanswerable questions score
  above the weakest answerable one. There is no cut that separates them, which is the finding,
  not a missing tuning step.
- It does technically beat always-answer on balanced accuracy, 0.556 against 0.500, and that
  is not a reason to ship it. The gain comes from refusing 8 of 18 answerable questions.
  Balanced accuracy weights both errors equally and they are not equal here: a false answer
  arrives with citations that resolve to exact character offsets and can be checked, a false
  abstention leaves nothing to check. Sensitivity floor of 0.90 declared in the script, not
  chosen after seeing the result.
- Reproduce with `python eval/abstention_threshold.py`; the result is pinned in
  `eval/abstention_threshold.json`.

## Generation

- **No answer-quality metric is published.** Only one local model is available, and grading
  its own output is circular.
- Generation is non-deterministic (default temperature) and is not part of the reference gate.
- **28% of answers cite a context block that was never supplied.** The prompt numbers five
  blocks and asks the model to cite the numbers it used. On the 18 answerable test
  questions it produced `[6]`, `[7]` and `[16]`: 5 of 18, **27.8% [12.5%, 50.9%]**. Every
  answer cited something, so the failure is not silence, it is a reference that points at
  nothing. Measured by `eval/citation_run.py`, pinned in `eval/citation_structure.json`.
  No judge is involved — these are facts about the string, which is why the circularity
  objection above does not reach them.
- **The `citations` field is retrieval, not attribution.** `answer()` returns
  `[citation(c) for c in contexts]`: all five retrieved blocks, whatever the answer used.
  Mean context coverage is **27.8%**, so most of the listed citations support nothing in
  the text they are attached to. The offsets resolve exactly, as claimed; what they do not
  establish is that the answer came from there.
- **The measurement is reproducible but not bit-exact.** Generation is pinned to
  temperature 0 with a fixed seed, which is not how the service runs and is declared in the
  artefact. Two independent runs agreed on 17 of 18 answers and on the headline 5 of 18;
  one question cited `[2]` rather than `[2, 4]`, moving mean coverage by 1.1 points. A
  local backend at temperature 0 is close to deterministic and is not guaranteed to be.
- **Stating the valid range helps, and 18 questions cannot establish that it does.** The
  served prompt says "cite the numbers you used" and never says which numbers exist. A
  variant differing by one clause — "the context blocks are numbered 1 to 5; cite only
  those numbers" — was run over the same questions and the same pinned retrieval:

  | | cite a block never given | mean context coverage |
  |---|---|---|
  | served | 5/18, 27.8% [12.5%, 50.9%] | 27.8% |
  | range stated | 2/18, 11.1% [3.1%, 32.8%] | 17.8% |

  Paired on the same questions it fixed three and broke none, which is the right
  direction; McNemar's exact test on three discordant pairs gives **p = 0.250**, and the
  two intervals overlap across most of their width. This is a hint, not a result.

  It also costs breadth. Coverage falls from 27.8% to 17.8% because the variant cites one
  block where the served prompt cited several, so part of the improvement is the model
  citing less rather than citing better.

  `nfip-016` and `nfip-024` emit `[6]` and `[7]` under both prompts, so two of the five
  failures are not the prompt's doing.

  **The served prompt is unchanged.** A variant ahead on eighteen questions at p = 0.250
  which also narrows what gets cited is not evidence enough to change what ships.
  Reproduce with `python eval/citation_run.py --variant ranged`; pinned in
  `eval/citation_structure_ranged.json`.
- **Whether a cited block supports its sentence is still not measured.** That is citation
  correctness, it needs an entailment model rather than the generator, and it is a
  different question from whether the citation is structurally real. Not attempted.

## Ingestion

- **Trailing duplicate chunk.** A document whose length lands just past a chunk boundary
  emits a final chunk wholly contained in the previous one. Retained deliberately: fixing it
  changes every chunk ID, which invalidates all 50 labels and the reference run. The benefit
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
