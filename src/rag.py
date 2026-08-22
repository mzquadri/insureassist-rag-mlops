"""
Retrieval + generation (the RAG core).

Retrieval is hybrid: dense BGE vectors from Qdrant fused with an in-process BM25 index by
reciprocal rank fusion. Both retrievers run over the same chunks with the same IDs, and the
architecture was selected from dev-split evidence recorded in `eval/retrieval_config.json`.

Every returned context is a *citation*: it carries the chunk ID, the document, the CFR
citation and the exact character offsets, so any claim can be traced back to committed
corpus text. Nothing here invents a span.

The embedder, the vector store, the lexical index and the generator all come from
`src.providers`, which builds them on first use. Importing this module costs nothing and
contacts nothing.
"""
from __future__ import annotations

from src.config import cfg
from src.errors import RetrievalUnavailable
from src.providers import (
    get_bm25_index,
    get_embedder,
    get_generator,
    get_vector_store,
)
from src.retrieval import Hit, reciprocal_rank_fusion

#: Candidates drawn from each retriever before fusion. From the frozen configuration.
CANDIDATE_DEPTH = 20

#: Answer statuses. `answered` means evidence was retrieved and a generator ran.
STATUS_ANSWERED = "answered"
STATUS_INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def _dense_candidates(question: str, limit: int) -> tuple[list[Hit], dict]:
    embedder = get_embedder()
    store = get_vector_store()

    # BGE is asymmetric: the query gets an instruction prefix, passages do not. Ingestion
    # embeds chunk text bare, so the prefix belongs here and only here.
    vector = embedder.encode(
        cfg.BGE_QUERY_PREFIX + question, normalize_embeddings=True
    ).tolist()
    try:
        result = store.query_points(
            collection_name=cfg.QDRANT_COLLECTION,
            query=vector,
            limit=limit,
            with_payload=True,
        )
    except Exception as exc:  # boundary: the store is external
        raise RetrievalUnavailable(
            f"query against collection {cfg.QDRANT_COLLECTION!r} failed"
        ) from exc

    payloads = {point.payload["chunk_id"]: point.payload for point in result.points}
    hits = [Hit(point.payload["chunk_id"], point.score) for point in result.points]
    return hits, payloads


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """Return the top-k most relevant chunks, fused from dense and lexical retrieval."""
    top_k = top_k or cfg.TOP_K

    dense_hits, payloads = _dense_candidates(question, CANDIDATE_DEPTH)
    lexical_hits = get_bm25_index().search(question, CANDIDATE_DEPTH)

    fused = reciprocal_rank_fusion([dense_hits, lexical_hits], limit=top_k)

    dense_scores = {h.chunk_id: h.score for h in dense_hits}
    lexical_scores = {h.chunk_id: h.score for h in lexical_hits}

    contexts = []
    for hit in fused:
        payload = payloads.get(hit.chunk_id)
        if payload is None:
            # BM25 surfaced a chunk the dense candidate list did not contain, so its
            # payload has not been fetched. Read it from the corpus rather than dropping
            # the hit or leaving a citation without offsets.
            payload = _payload_from_corpus(hit.chunk_id)
        if payload is None:
            continue
        contexts.append({
            "chunk_id": payload["chunk_id"],
            "document_id": payload["document_id"],
            "source": payload.get("source", payload["document_id"]),
            "form": payload.get("form", ""),
            "cfr_citation": payload.get("cfr_citation", ""),
            "start": payload["start"],
            "end": payload["end"],
            "text": payload["text"],
            "score": hit.score,
            "dense_score": dense_scores.get(hit.chunk_id),
            "lexical_score": lexical_scores.get(hit.chunk_id),
        })
    return contexts


_CORPUS_PAYLOADS: dict[str, dict] | None = None


def _payload_from_corpus(chunk_id: str) -> dict | None:
    """Payload for a chunk the dense list did not return, read from committed corpus."""
    global _CORPUS_PAYLOADS
    if _CORPUS_PAYLOADS is None:
        from src.corpus import chunk_corpus, load_corpus

        documents = load_corpus()
        by_id = {d.document_id: d for d in documents}
        _CORPUS_PAYLOADS = {
            chunk.chunk_id: chunk.payload(by_id[chunk.document_id])
            for chunk in chunk_corpus(documents, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP)
        }
    return _CORPUS_PAYLOADS.get(chunk_id)


def reset_corpus_cache() -> None:
    """Drop the cached corpus payloads. Tests use this; ingestion changes invalidate it."""
    global _CORPUS_PAYLOADS
    _CORPUS_PAYLOADS = None


def build_prompt(question: str, contexts: list[dict]) -> str:
    context_block = "\n\n".join(
        f"[{index}] Source: {c['source']} ({c['cfr_citation']})\n{c['text']}"
        for index, c in enumerate(contexts, start=1)
    )
    return (
        "You are an insurance policy assistant. Answer the question using ONLY the "
        "context below. Each context block is numbered; cite the numbers you used in "
        "square brackets. If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def generate(prompt: str) -> str:
    """Generate an answer with whichever backend `LLM_BACKEND` selects."""
    return get_generator()(prompt)


def citation(context: dict, excerpt_chars: int = 240) -> dict:
    """A traceable citation for one retrieved chunk.

    Offsets index the committed corpus file, so `document.text[start:end]` reproduces the
    cited chunk exactly. The excerpt is a prefix of that text - never a paraphrase and
    never generated.
    """
    excerpt = context["text"][:excerpt_chars]
    if len(context["text"]) > excerpt_chars:
        excerpt = excerpt.rstrip() + "..."
    return {
        "chunk_id": context["chunk_id"],
        "document_id": context["document_id"],
        "source": context["source"],
        "form": context["form"],
        "cfr_citation": context["cfr_citation"],
        "start": context["start"],
        "end": context["end"],
        "score": round(context["score"], 6),
        "excerpt": excerpt,
    }


def answer(question: str) -> dict:
    """Full RAG: retrieve -> prompt -> generate. Returns a status, an answer and citations.

    `status` is structured rather than implied by prose. If retrieval returns nothing there
    is no evidence to answer from, so the service says so instead of asking a generator to
    improvise - that is the one abstention case the dev evidence actually supports. See
    docs/EVALUATION.md for why no similarity threshold is applied on top of this.
    """
    contexts = retrieve(question)

    if not contexts:
        return {
            "status": STATUS_INSUFFICIENT_EVIDENCE,
            "answer": (
                "No policy text was retrieved for this question, so it cannot be answered "
                "from the indexed documents."
            ),
            "citations": [],
        }

    text = generate(build_prompt(question, contexts))
    return {
        "status": STATUS_ANSWERED,
        "answer": text,
        "citations": [citation(c) for c in contexts],
    }


if __name__ == "__main__":
    import json

    print(json.dumps(answer("What is the Increased Cost of Compliance limit?"), indent=2))
