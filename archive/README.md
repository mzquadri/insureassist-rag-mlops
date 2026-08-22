# Archive

Work that was authored but is **not part of the product**, kept for honesty rather than
deleted. Nothing here is referenced by the service, the benchmark, or the reference run.

## `finetune/` — LoRA fine-tuning notebook

A Colab recipe adapting Phi-3-mini with PEFT and logging to MLflow.

**Archived rather than executed.** Two independent reasons:

1. **No evidence it ever ran.** All sixteen cells are committed with no output and no
   execution count. No adapter was produced, and no MLflow run data exists.
2. **Its training data is the old evaluation set.** The ten training pairs are the ten
   questions from `data/qa_testset.jsonl`. Training set and test set were the same set, so
   no number from that harness could ever have described a fine-tuned model.

Executing it properly would need a separate dataset with real train/validation/test
isolation and no overlap with the NFIP benchmark. That is a project of its own, and the
flagship does not depend on it: the served system uses a stock model, and every published
result is a *retrieval* result that a fine-tuned generator would not change.

It stays here because deleting it would erase the leakage finding along with the code.
