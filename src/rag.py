"""
Phase 1 — Retrieval + Generation (the "RAG" core).

1. Embed the user's question.
2. Search Qdrant for the most similar document chunks (retrieval).
3. Put those chunks into a prompt and ask an LLM to answer using ONLY that context.

The generator is pluggable: Ollama now (Phase 1), the fine-tuned HF model later (Phase 3).
"""
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.config import cfg

# Load once at import time (heavy objects)
_embedder = SentenceTransformer(cfg.EMBEDDING_MODEL)
_client = QdrantClient(url=cfg.QDRANT_URL)


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """Return the top-k most relevant chunks for a question."""
    top_k = top_k or cfg.TOP_K
    qvec = _embedder.encode(question, normalize_embeddings=True).tolist()
    hits = _client.search(
        collection_name=cfg.QDRANT_COLLECTION,
        query_vector=qvec,
        limit=top_k,
    )
    return [
        {"text": h.payload["text"], "source": h.payload["source"], "score": h.score}
        for h in hits
    ]


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in contexts
    )
    return (
        "You are an insurance policy assistant. Answer the question using ONLY the "
        "context below. If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def generate_ollama(prompt: str) -> str:
    """Call a local Ollama model (Phase 1)."""
    resp = requests.post(
        f"{cfg.OLLAMA_URL}/api/generate",
        json={"model": cfg.OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def generate(prompt: str) -> str:
    if cfg.LLM_BACKEND == "ollama":
        return generate_ollama(prompt)
    elif cfg.LLM_BACKEND == "hf":
        # Implemented in Phase 3 (loads base model + LoRA adapter)
        from src.hf_generator import generate_hf
        return generate_hf(prompt)
    raise ValueError(f"Unknown LLM_BACKEND: {cfg.LLM_BACKEND}")


def answer(question: str) -> dict:
    """Full RAG: retrieve -> prompt -> generate. Returns answer + sources."""
    contexts = retrieve(question)
    prompt = build_prompt(question, contexts)
    text = generate(prompt)
    return {
        "answer": text,
        "sources": [{"source": c["source"], "score": round(c["score"], 3)} for c in contexts],
    }


if __name__ == "__main__":
    import json
    q = "Does home insurance cover water damage from a burst pipe?"
    print(json.dumps(answer(q), indent=2))
