# Architecture

```
data/corpus/*.txt  (NFIP forms, SHA-256 verified against manifest.json)
  |
  +- src/corpus.py -- deterministic chunking (800 chars, 120 overlap)
  |                   chunk_id = "{document_id}#{sha256(document_id|start|text)[:12]}"
  |                   point_id = uuid5(namespace, chunk_id)
  |
  +- src/ingest.py -- BGE embeddings --> Qdrant (cosine, 384-dim)
  |                   idempotent: re-ingest overwrites the same 314 points
  |
  +- query time (src/rag.py)
       +- dense:   BGE query prefix + cosine search, 20 candidates
       +- lexical: in-process Okapi BM25, 20 candidates
       +- fusion:  reciprocal rank fusion (k=60) --> top 5
       +- citations: chunk_id, document, CFR citation, start/end offsets, excerpt
       +- generation: local Ollama model, prompted to cite context numbers
```

## Why each piece is there

**Hybrid, not dense.** Dense and BM25 each found 3 dev questions the other missed (union
0.857 against 0.643 for either alone). Neither is defensible when the other holds a fifth of
the answers.

**RRF, not score blending.** Cosine similarity and BM25 scores are on different unnormalised
scales. Any weighted sum invents a calibration; RRF reads rank positions only.

**BM25 in-process, not in Qdrant.** Three documents, 314 chunks: the index builds in
milliseconds and needs no second service.

**Content-derived chunk IDs.** Ground-truth labels name chunk IDs. With random UUIDs a label
stopped pointing at its evidence after every re-ingest, which made labelled evaluation
impossible.

**Document ID inside the chunk hash.** The three forms contain word-for-word identical
passages; hashing text alone would collide across documents and merge distinct chunks.

**Lazy providers.** `src/providers.py` builds the embedder, vector store, lexical index and
generator on first use and lets tests replace them. That is why `import src.api` needs no
network and why the suite runs offline in about two seconds.

**Split probes.** `/health` is liveness and touches nothing. `/ready` checks every serving
dependency. A single endpoint for both meant pods were marked ready while Qdrant was down.

## What is deliberately absent

**No reranker.** The dev failure mode is candidate selection: the right chunk is often absent
from the list entirely. A reranker cannot promote what was never retrieved.

**No query expansion.** It would blur the form distinction the benchmark turns on.

**No metadata form filter.** Inferring the target form from query wording is feasible, but
fusion already lifts top-document accuracy to parity with the best single retriever, and a
filter that guesses wrong removes the correct document outright.

**No abstention threshold.** Not defensible on this data; see [EVALUATION.md](EVALUATION.md).

**No fine-tuning.** Archived. Every published result is a retrieval result.
