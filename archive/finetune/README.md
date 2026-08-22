# LoRA fine-tuning (Google Colab)

> **Status: authored, not evidenced.**
>
> This notebook is complete and internally coherent, but **the repository contains no proof
> that it was ever run to completion**. Every cell is committed with no output and no
> execution count, no adapter is present, and no MLflow run data is stored. The served
> system does not use a fine-tuned model; it calls a stock model through Ollama.
>
> Treat what follows as a training *recipe*, not as a result.

## The training data cannot evaluate this model

The ten `(question, answer)` pairs in the notebook are **the same ten questions** as the
evaluation set in `data/qa_testset.jsonl`, with near-identical reference answers.

That means the training set and the test set are the same set. A model tuned on this data
would be scored on examples it had memorised, so:

- **no number produced by `eval/evaluate.py` can describe a fine-tuned model.** With ten
  examples over ten epochs, memorisation is the expected outcome, not a risk.
- there is **no validation split** in the notebook at all - `Trainer` receives only
  `train_dataset`.
- fixing this needs held-out questions written against documents the model never saw, not
  a re-run of the same notebook.

This is disclosed rather than quietly corrected because the overlap is the reason the
fine-tuned path carries no published metric.

## Why Colab?

Fine-tuning needs a GPU, and Colab provides a free T4.

## Steps

1. Upload `lora_finetune.ipynb` to https://colab.research.google.com.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Run every cell top to bottom.
4. Download the `adapter/` folder into `finetune/adapter/` (git-ignored: it is a build
   artifact, not source).

## What the notebook actually does

- **LoRA** on Phi-3-mini: `r=16`, `alpha=32`, `dropout=0.05`, targeting `qkv_proj` and
  `o_proj`. The base model stays frozen; only the adapter trains.
- **fp16 with the standard Hugging Face `Trainer`.** Earlier drafts used TRL and 4-bit
  bitsandbytes quantisation; both were removed during Phase 2 to get past Colab dependency
  conflicts, so neither QLoRA nor TRL is involved despite what earlier notes claimed.
- **MLflow** logs parameters, the final training loss, and the adapter as an artifact.
  Note that `report_to=[]` disables the `Trainer`'s own MLflow integration, so there is no
  per-step loss curve, and with no tracking URI configured the run is written to the Colab
  VM's local `./mlruns` - which disappears when the runtime ends unless it is downloaded.

## Using an adapter, if you train one

Set `LLM_BACKEND=hf`. `src/hf_generator.py` loads the base model and attaches the adapter
at `LORA_ADAPTER_PATH` if that directory exists, and silently uses the base model if it
does not. That path also needs `torch`, `transformers` and `peft`, which are **not** in
the root `requirements.txt`.
