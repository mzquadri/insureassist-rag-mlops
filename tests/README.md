# Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

The whole suite runs in about a second. **Every test is offline**: none downloads a model,
opens a socket, contacts Qdrant, or needs Ollama or Docker.

That is a property of the design, not a convention. `src/providers.py` builds the embedder,
the vector store and the generator on first use and lets a test replace any of them, so the
application can be imported and exercised without its dependencies. Two tests defend it
directly: one asserts that importing `src.api` leaves `sentence_transformers`, `torch` and
`qdrant_client` out of `sys.modules`; the other imports the app in a subprocess where every
socket operation raises.

## What is covered

| Area | File | Notes |
|---|---|---|
| Chunking behaviour and boundaries | `test_chunking.py` | Includes the corpus-size claim used in the README |
| Rejected chunk configurations | `test_chunking.py` | `overlap >= size` used to hang forever |
| Import is inert / network-free | `test_api_offline.py` | The load-bearing guarantee |
| `/health` liveness | `test_api_offline.py` | Answers with no providers configured at all |
| `/ask` happy path | `test_api_offline.py` | Injected fake retriever and generator |
| `/ask` request validation | `test_api_offline.py` | Empty, missing, oversized, wrong type |
| Dependency failure mapping | `test_api_offline.py` | Qdrant / embedder / generator down → 503 |
| Provider seam | `test_providers_and_rag.py` | Overrides, reset, backend resolution |
| Retrieval and prompt building | `test_providers_and_rag.py` | Error wrapping preserves `__cause__` |
| Judge parsing and NaN accounting | `test_judge_scoring.py` | Refusals must become NaN, never a number |
| Recorded report reconciliation | `test_judge_scoring.py` | Published averages recomputed from the CSV |
| Ingestion determinism | `test_ingest_determinism.py` | Scaffold; one strict xfail marks the open gap |

## The expected xfail

`test_chunk_ids_are_derived_from_content` is a **strict xfail**. Chunk IDs are currently
random UUIDs, so the same corpus ingested twice yields different identifiers, and a
relevance label cannot name a chunk that survives re-ingestion. When content-derived IDs
land, this test starts passing and pytest will fail the run until it is finished properly.

## What is deliberately not here

No test starts Docker, Qdrant, or Ollama. Integration coverage against a real vector store,
and a container test that makes a real HTTP request, are a later phase — they belong in CI
with service containers rather than in a suite that has to stay fast and offline.
