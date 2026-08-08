"""
Phase 3 — Generate answers with the fine-tuned Hugging Face model (base + LoRA adapter).

Set LLM_BACKEND=hf in .env to use this instead of Ollama. It loads the base model once,
attaches your LoRA adapter from finetune/adapter/, and generates text.

Note: on a CPU-only laptop this is slow; it's mainly for the cloud/GPU deployment.
For quick local dev keep LLM_BACKEND=ollama.
"""
from functools import lru_cache

from src.config import cfg


@lru_cache(maxsize=1)
def _load():
    """Load base model + LoRA adapter once and cache it."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(cfg.HF_BASE_MODEL, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        cfg.HF_BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    # Attach the fine-tuned adapter (skip gracefully if not present yet)
    import os
    if os.path.isdir(cfg.LORA_ADAPTER_PATH):
        model = PeftModel.from_pretrained(base, cfg.LORA_ADAPTER_PATH)
    else:
        print(f"[hf_generator] No adapter at {cfg.LORA_ADAPTER_PATH}; using base model.")
        model = base
    model.eval()
    return tokenizer, model


def generate_hf(prompt: str, max_new_tokens: int = 256) -> str:
    import torch
    tokenizer, model = _load()
    messages = [
        {"role": "system", "content": "You are a precise insurance policy assistant."},
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
    return text.strip()
