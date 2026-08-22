# Evaluation methodology

## Splits

40 hand-authored questions, split by a hash of the question ID stratified by category:
**dev 18, test 22**. Deterministic, and asserted by a test.

Dev drove every decision: architecture, chunk size, fusion depth, BM25 parameters, and the
abstention investigation. **Test was run once**, after the configuration was frozen into
`eval/retrieval_config.json`. No tuning followed it.

The cost of that discipline is visible: hybrid scored hit@5 **1.000** on dev and **0.556** on
test. Selecting on 14 answerable questions overfits, and the gap is reported rather than
hidden.

## Labels

Questions, target forms, categories and gold answers are hand-authored. Chunk IDs are not:
each label carries a regex anchor, and `scripts/build_ground_truth.py` resolves it to offsets
and covering chunks. Hand-transcribing hashes produces labels that point at the wrong
passage, which makes a metric silently wrong rather than visibly broken.

An anchor matching zero times, more than once, or spanning a chunk boundary is an error.
Because labels regenerate from anchors, a chunking change re-derives them correctly instead
of invalidating the benchmark.

`python -m eval.validate_ground_truth` rejects broken references, spans outside a document,
answerable questions without evidence, unanswerable questions carrying evidence, duplicates,
and missing provenance.

## Metrics

Pure functions in `eval/metrics.py`, tested against hand-computed values.

Two decisions that change results silently if got wrong: duplicate retrieved IDs are
collapsed (a retriever must not score two hits for one chunk), and an empty relevance set
returns **NaN, not zero** - an unanswerable question has no recall, and a fabricated zero
would look like a genuine failure.

**nDCG is not implemented.** Labels are binary; there is no basis for calling one chunk twice
as relevant as another. With one relevant chunk it would be a monotone function of the
reciprocal rank already reported.

## Citations, deterministic

Citation precision and recall are computed from retrieval against the labelled chunks. No
model is involved.

`unsupported_citation_rate` checks that every citation's offsets reproduce its quoted excerpt
from the committed corpus. It is **0.000** and must stay there: a non-zero value means the
service is fabricating provenance.

Precision is bounded by construction - one relevant chunk out of five returned caps it at 0.2.

## Abstention, investigated and not claimed

Four dev signals were measured: top-1 dense score, top-1 RRF score, top-1 minus top-2 margin,
and dense/lexical consensus.

| Signal | Best single-threshold accuracy | Always-answer baseline |
|---|---|---|
| top-1 dense | 0.778 | 0.778 |
| top-1 RRF | 0.833 | 0.778 |
| margin | 0.833 | 0.778 |
| consensus | 0.833 | 0.778 |

The best signals beat the baseline by exactly one question out of eighteen, with four
unanswerable questions in dev. That is noise, not a finding.

The decisive case: `nfip-034` asks for a coinsurance threshold under a form that has no
coinsurance article. Its top-1 dense score is **0.787**, above the answerable mean of
**0.761**. Any threshold catching it would reject most real questions.

**So no threshold is implemented.** Only the structural case fires: retrieval returns nothing
gives `insufficient_evidence`. On this corpus that never happens, so unanswerable rejection
is 0.000. That is a real weakness and is reported as one.

## Answer quality, not published

No judge metrics. Only one local model is installed, and using it to grade its own output is
precisely the circular evaluation this project removed earlier. A judge must be a different
model; downloading one purely to populate the field would not make the number trustworthy.

The earlier custom judge harness (`eval/evaluate.py`) still exists and is still honestly
labelled `judge_*`. It is **not** RAGAS and never was, and it is not part of the reference
gate.

## Leakage

The original 10-question test set was byte-for-byte the fine-tuning training data. That is
why fine-tuning is archived and why no tuned-model number appears anywhere.

The NFIP benchmark shares no text with it, and a test asserts that no benchmark question
appears in the archived notebook.
