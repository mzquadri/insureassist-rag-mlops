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
- **Citations are frequently redundant, and support is uncertain.** This is citation
  correctness: ALCE's recall and precision (Gao et al., arXiv:2305.14627), judged by an
  entailment model rather than by the generator, which is the standard answer to the
  circularity above. `eval/citation_correctness.py` implements the metric and
  `eval/citation_correctness_run.py` runs it with `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`.

  | metric | count | rate | 95% Wilson |
  |---|---|---|---|
  | citation recall | 7/15 scoreable sentences supported | 0.4667 | [0.248, 0.699] |
  | citation precision | 5/15 citations needed | 0.3333 | [0.152, 0.583] |

  Over all 16 cited sentences, including the one with no content to score, recall is
  7/16 = 0.4375 [0.231, 0.668]. Both are in `eval/citation_correctness.json`; the
  scoreable denominator is the one quoted, for the reason in the next paragraph.

  **What this does and does not establish.** Precision is the firmer of the two: its upper
  bound is 0.583, so even on the most favourable reading consistent with the data, at least
  two citations in five were not needed — drop them and the rest still support the sentence.
  Recall is genuinely uncertain. Its interval spans 0.248 to 0.699, which straddles a half,
  so the honest statement is that somewhere between a quarter and seven-tenths of cited
  sentences are supported and fifteen sentences cannot narrow it further. An earlier version
  of this entry claimed recall's ceiling put support below three in five; that rested on a
  figure since corrected, and the claim is withdrawn.

  **A quarter of the published recall figure was marker-removal debris.** The first version
  of this entry reported 5/16 = 0.3125. That was wrong, and the fault was in extracting the
  claim rather than in the judge. This generator places its citations in subject position —
  the raw answer to `nfip-002` is literally `"[2], [5]\n\nThe maximum payable under Coverage
  D ... is $30,000."` — so stripping the markers left a comma and a paragraph break glued to
  the front of the sentence. Nine of sixteen hypotheses were malformed that way and one was
  the empty string, which no premise can entail. Collapsing the whitespace and stripping the
  stranded punctuation moved recall from 0.3125 to 0.4667 and precision from 0.4444 to
  0.3333. The judge was never the problem here; the text handed to it was.

  Exactly 1 sentence is excluded from the scoreable denominator: after removing its
  markers nothing remained, so there is no statement for a person or a model to rule on, and
  `eval/citation_structure.py` already reports it as a structural failure. Fragments are
  *not* excluded. An intermediate version of this fix discarded any hypothesis beginning
  with a conjunction, which threw away `"and According to the context, ... the revised due
  date will be 30 days after the date on which the bill is mailed"` — a real claim with a
  connective stuck to the front. Guessing at grammar with a regex is how a measurement
  quietly loses its evidence; the conjunction is now stripped as debris and the sentence is
  kept.

  **Four of the fifteen sentences are about the retrieval, not about flood policy.**
  `"provide relevant information"`, `"were used to answer the question"`, `"do not provide a
  clear answer to this question"`. Their grammatical subject is the retrieved blocks, so no
  passage of policy text can entail them and they count against recall for a reason that has
  nothing to do with citation quality. Nine of the fifteen open with `"According to ..."`.
  That the generator narrates its own retrieval this often is a finding about the generator,
  and it is a confound in this metric rather than a result of it.

  **The calibration is on a proxy population, not this one.** This is the sharpest caveat
  and it was missing from the first version of this entry. The judge is calibrated on
  *(chunk, gold_answer)* pairs; the rates are computed on *(concatenated cited blocks,
  generated sentence)* pairs. The premise is built differently and the hypothesis comes
  from a different writer, so the 0.844 sensitivity does not transfer to the measured
  pairs, and no correction for judge error can be read off it. An earlier version of this
  entry used it to argue the rate was "biased low"; that inference is withdrawn — the
  direction of the judge's error on the pairs actually scored is unmeasured.

  This also closes off the obvious statistical repair. Prediction-powered inference would
  debias a machine-labelled rate using a labelled subset, but its validity rests on that
  subset being drawn from the target population, and this one is not.

  **A human-labelled gold standard is the fix, and the worksheet exists.** ALCE validates
  its own judge the same way — 100 human-annotated examples per dataset — and that step has
  never been done here. At fifteen scoreable sentences, labelling every pair by hand costs
  less than any statistical correction and settles both open questions at once: what the
  rate actually is, and whether this judge agrees with a person *on the pairs that produced
  the number*. `python -m eval.citation_gold_run --build` writes
  `eval/citation_gold/worksheet.md` — 15 support judgments and 21 drop-one necessity
  judgments, each showing the exact premise the metric scored — and `--score` reports the
  human rates alongside the judge's sensitivity, specificity and Cohen's kappa against them.

  **The labels are not in yet, so no human rate is published.** `eval/citation_gold.json`
  appears only once `labels.jsonl` is filled in, and `human_counts` returns `None` until
  every support judgment is answered: a rate over half the sentences is a different quantity
  wearing the same name. A pre-seeded `null` is treated as unanswered rather than as `false`,
  so an untouched worksheet cannot be mistaken for a corpus of unsupported citations.
- **The generator puts its citations before the claim, and no prompt tried here changes
  that.** The `[n]` convention, and ALCE's metric built on it, assume the marker *follows*
  the statement it supports — that is what makes "which claim does this citation back"
  answerable at all. This model does the opposite. No judge is involved: whether a sentence
  opens with a marker is a fact about the string, which is why this escapes the circularity
  objection above for the same reason the dangling-citation rate does.

  | | opens with a citation | citations joined by a conjunction |
  |---|---|---|
  | served | 12/16, 0.75 [0.505, 0.898] | 6/16, 0.375 [0.185, 0.614] |
  | range stated | 16/16, 1.0 [0.806, 1.0] | 0/16, 0.0 [0.0, 0.194] |

  **Placement is systematic, not a prompt artefact.** Paired on the 13 questions that cited
  something under both prompts, 12 open with a citation under *both*, none under neither,
  and the single discordant pair goes to the range-stated prompt: McNemar's exact test gives
  p = 1.0. Restating which block numbers exist does not move where the model puts them.

  **Stating the range does remove coordinated citations, and this is the clearest prompt
  effect measured in this repository.** Five questions emit `"[2], [3], and [4] provide
  relevant information"` under the served prompt and none do under the variant, with no pair
  going the other way: p = 0.0625. That is still above 0.05 on thirteen questions, so it is
  a strong hint rather than a result — but it is four times better evidence than the
  p = 0.250 for the dangling-citation comparison above, and it points the same way.

  **This is what broke the correctness measurement**, which is why it is worth a metric of
  its own rather than a footnote. Leading markers are what stranded punctuation at the front
  of nine of sixteen hypotheses and emptied one of them entirely, and repairing that moved
  recall from 0.3125 to 0.4667. A defect that silently cost a quarter of a published figure
  is not cosmetic.

  It also weakens a promise the README makes carefully. Citations resolve to exact character
  offsets, and they do; what offsets cannot do is say which sentence they belong to when the
  marker precedes every claim in the answer. Three of the five excluded questions
  (`nfip-004`, `nfip-005`, `nfip-028`) cited nothing at all under the served prompt, so they
  are left out of the pairing — an answer with no citations has not placed them well, and
  counting it as a win would reward silence.

  Reproduce with `python -m eval.citation_placement_run`; pinned in
  `eval/citation_placement.json`. The exact McNemar test is now in
  `eval/citation_placement.py` and pinned by a test against the p = 0.250 quoted above,
  which until now had been computed by hand.

  ALCE's own agreement figures set the ceiling on what this arm can claim even when the
  judge is right: on 100 human-annotated examples per dataset they report Cohen's kappa
  0.698 for citation recall and **0.525 for citation precision** — only moderate. The
  precision figure above rests on the weaker of the two metrics, measured here by a much
  smaller judge than the T5-11B those kappas describe.

  Correctness is also not faithfulness: a block supporting a claim does not establish the
  model derived the claim from it.

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
