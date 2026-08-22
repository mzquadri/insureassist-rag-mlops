# Data provenance

Everything in this directory is **synthetic and self-authored**. None of it is a real
insurance policy, and none of it was copied from an insurer, a regulator, or any other
publisher. It was written by hand for this project so that the repository carries no
third-party licensing obligation.

| File | What it is | Origin | Licence |
|---|---|---|---|
| `auto_insurance_policy.md` | Invented summary of motor cover (1,486 characters) | Written by hand for this project | MIT, with the rest of the repository |
| `home_insurance_policy.md` | Invented summary of home cover (1,565 characters) | Written by hand for this project | MIT, with the rest of the repository |
| `qa_testset.jsonl` | 10 questions with reference answers | Written by hand from the two documents above | MIT, with the rest of the repository |

No personal, customer, or confidential information appears in any of these files. Clause
wording, monetary limits, and time periods are invented and resemble common market terms
only in the general way any plain-language policy summary would.

## What this corpus can and cannot support

At the configured chunk size of 600 characters with 100 characters of overlap, the two
documents produce **7 chunks in total**. Retrieval runs with `TOP_K=4`.

Four of seven chunks is **more than half the entire index on every single query**. Recall
at that setting is close to meaningless: almost any question will have its evidence inside
the returned window regardless of how good or bad the retriever is. This is the direct
cause of the flat `context_precision` scores discussed in `eval/README.md`.

So this corpus is adequate for:

- exercising the pipeline end to end,
- unit and integration test fixtures,
- an offline demonstration that does not need network access.

It is **not** adequate for:

- measuring retrieval quality,
- comparing retrievers or chunking strategies,
- any published claim about how well the system finds evidence.

A larger corpus of genuinely public, verifiably redistributable documents is required
before retrieval numbers mean anything. That work is scoped but not yet done; until it
lands, no retrieval metric is published anywhere in this repository.

## Overlap with the fine-tuning data

The ten questions in `qa_testset.jsonl` are the same ten questions used as **training
data** in `archive/finetune/lora_finetune.ipynb`. See `archive/finetune/README.md` for what that rules out.
