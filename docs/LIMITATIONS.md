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
- **Most cited sentences are not supported by the block they cite.** This is citation
  correctness: ALCE's recall and precision (Gao et al., arXiv:2305.14627), judged by an
  entailment model rather than by the generator, which is the standard answer to the
  circularity above. `eval/citation_correctness.py` implements the metric and
  `eval/citation_correctness_run.py` runs it with `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`.

  | metric | count | rate | 95% Wilson |
  |---|---|---|---|
  | citation recall | 5/16 cited sentences supported | 0.3125 | [0.142, 0.556] |
  | citation precision | 4/9 citations needed | 0.4444 | [0.189, 0.733] |

  **What this does and does not establish.** Sixteen cited sentences is a small
  denominator and the intervals are correspondingly wide, so the rate itself is not pinned.
  What the interval does exclude is the comfortable reading: recall's upper bound is 0.556,
  so even on the most generous reading consistent with the data, fewer than three in five
  cited sentences are supported by what they cite. Combined with the 27.8% dangling-citation
  rate above, a citation in this system is weak evidence that the answer rests on the
  passage named.

  The judge's own sensitivity of 0.844 means it misses roughly one supported pair in six,
  so 0.3125 is biased low rather than high; the direction of the error does not rescue the
  figure, it widens it upward. Correctness is also not faithfulness: a block supporting a
  claim does not establish the model derived the claim from it.

  **The judge is calibrated against the ground truth before any rate is read.** Positives
  pair a human-written `gold_answer` with the chunk containing its labelled evidence
  offsets; negatives pair the same answer with a chunk the labels do not mark relevant. The
  cut is chosen by Youden's J, the method `eval/abstention_threshold.py` already uses here.

  | pairs | threshold | sensitivity | specificity | balanced accuracy |
  |---|---|---|---|---|
  | 32 / 32 | 0.608 | 0.844 | 0.969 | 0.906 |

  Both floors — 0.80 sensitivity and 0.80 specificity — were fixed in the script before the
  run, following `eval/abstention_threshold.py`, and the artefact carries `reportable:
  false` when either is missed. Three earlier attempts missed them, and each failure was a
  fault in the measurement rather than a reason to lower the bar:

  | attempt | control | outcome |
  |---|---|---|
  | argmax on whole 800-char blocks | 16/30 verbatim entailed | judge under-entails; no rate |
  | same, larger judge | 15/30 *foreign* pairs entailed | judge over-entails; no rate |
  | Youden cut on verbatim positives | 30/30, 1/30 | cut lands at 0.98; rejects paraphrase |

  The first row is history and cannot be reproduced from the current code, which no longer
  decides by argmax. The second is still in `eval/citation_correctness_large.json` and is
  pinned from there; the third is still run on every invocation as a second check.

  The first failure was diagnosed, not assumed: much of this corpus is exclusion schedules
  whose entries — "Food freezers, other than walk-in, and food in any freezer." — end in a
  full stop and assert nothing standing alone, and the premise was ten times the length the
  model was trained on. The fix is to score each premise sentence separately and take the
  best, which is "retrieve-and-classify" from *Stretching Sentence-pair NLI Models to Reason
  over Long Documents* (arXiv:2204.07447); it moves the premise **towards** ALCE's
  granularity, which segments corpora into 100-word passages, rather than away from it.

  The third failure is the subtler one and is worth stating plainly. A control built from
  sentences copied verbatim out of their premise separates almost perfectly, which looks
  like success, and calibrates the wrong task: copies score near 1.0, so Youden's J put the
  cut at 0.98, and a citation metric never scores copies — an answer paraphrases its source.
  Applying that cut halved recall to 0.125. A control has to be *representative* as well as
  discriminative. The verbatim control is still run and recorded, as a second check in
  `judge_control_verbatim`, but it no longer sets the threshold.

  Reproduce with `python eval/citation_correctness_run.py --judge base|large`. Counts as
  well as rates are recorded in `eval/citation_correctness.json`, so the intervals above can
  be recomputed without re-running the model.

  Still open: ALCE validated against TRUE (T5-11B) at Cohen's kappa 0.698, which does not
  run on the 2 GiB card this was measured on, so that agreement is not inherited. And 3 of
  18 answers produced no scoreable sentence at all, because their only citations were out of
  range.

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
