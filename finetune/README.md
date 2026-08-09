# Phase 2 — LoRA fine-tuning (on Google Colab)

## Why Colab?
Fine-tuning needs a GPU. Google Colab gives a **free T4 GPU**. Your laptop's Intel GPU
can also work but the tooling (PEFT/TRL/bitsandbytes) is smoother on Colab, so we start there.

## Steps
1. Go to https://colab.research.google.com → **File → Upload notebook** → choose
   `lora_finetune.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Run every cell top to bottom (Runtime → Run all).
4. When done, download the `adapter/` folder and copy it into this repo at
   `finetune/adapter/` (it's git-ignored — too large/personal to commit).

## What you get
- A LoRA **adapter** (a few MB) that adapts Phi-3-mini to insurance answers.
- MLflow logs (params, loss, adapter artifact) — proof of experiment tracking.

## Concepts
- **LoRA**: train small low-rank matrices added to attention layers; base model frozen.
- **4-bit quantization (bitsandbytes)**: store the base model compactly to fit the GPU.
- **SFT (Supervised Fine-Tuning)**: train on (instruction, answer) pairs.
- **MLflow**: track and compare runs; register the best adapter.

Next: Phase 3 uses this adapter in `src/hf_generator.py`.
