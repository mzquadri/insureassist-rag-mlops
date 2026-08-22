"""
Retrieval + generation (the RAG core).

1. Embed the user's question.
2. Search Qdrant for the most similar document chunks (retrieval).
3. Put those chunks into a prompt and ask an LLM to answer using ONLY that context.

The embedder, the vector store, and the generator all come from `src.providers`, which
builds them on first use. Importing this module therefore costs nothing and contacts
nothing; see that module for why that matters.
"""
from src.config import cfg
from src.errors import RetrievalUnavailable
from src.providers import get_embedder, get_generator, get_vector_store


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """Return the top-k most relevant chunks for a question."""
    top_k = top_k or cfg.TOP_K
    embedder = get_embedder()
    store = get_vector_store()

    # BGE is asymmetric: the query gets an instruction prefix, passages do not. Ingestion
    # embeds chunk text bare, so the prefix belongs here and only here.
    qvec = embedder.encode(
        cfg.BGE_QUERY_PREFIX + question, normalize_embeddings=True
    ).tolist()
    try:
        result = store.query_points(
            collection_name=cfg.QDRANT_COLLECTION,
            query=qvec,
            limit=top_k,
            with_payload=True,
        )
    except Exception as exc:  # boundary: the store is external
        raise RetrievalUnavailable(
            f"query against collection {cfg.QDRANT_COLLECTION!r} failed"
        ) from exc

    return [
        {"text": h.payload["text"], "source": h.payload["source"], "score": h.score}
        for h in result.points
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


def generate(prompt: str) -> str:
    """Generate an answer with whichever backend `LLM_BACKEND` selects."""
    return get_generator()(prompt)


def answer(question: str) -> dict:
    """Full RAG: retrieve -> prompt -> generate. Returns answer + sources.

    `sources` identifies the *documents* the answer drew on, not the exact passages. That
    is document-level attribution rather than a citation, and it is not yet verifiable;
    chunk-level citation is a later phase.
    """
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
