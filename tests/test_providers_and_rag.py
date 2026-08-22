"""The dependency seam and the retrieval path, without a model or a database."""
import pytest

from src import providers
from src.config import cfg
from src.errors import GenerationUnavailable, RetrievalUnavailable
from src.rag import STATUS_ANSWERED, answer, build_prompt, citation, generate, retrieve


class TestProviderSeam:
    def test_overrides_are_returned(self, fake_embedder, fake_store):
        providers.set_embedder(fake_embedder)
        providers.set_vector_store(fake_store)
        assert providers.get_embedder() is fake_embedder
        assert providers.get_vector_store() is fake_store

    def test_reset_clears_overrides(self, fake_embedder):
        providers.set_embedder(fake_embedder)
        providers.reset()
        assert providers._embedder is None

    def test_generator_override_is_used(self):
        providers.set_generator(lambda prompt: "stubbed")
        assert generate("anything") == "stubbed"

    def test_unknown_backend_is_a_configuration_error(self, monkeypatch):
        """A typo in LLM_BACKEND is a deployment mistake, not a transient outage.

        It must therefore raise ValueError rather than a DependencyUnavailable that the API
        would translate into a retryable 503.
        """
        monkeypatch.setattr(cfg, "LLM_BACKEND", "not-a-backend")
        providers.reset()
        with pytest.raises(ValueError, match="Unknown LLM_BACKEND"):
            providers.get_generator()

    def test_known_backends_resolve_without_being_called(self, monkeypatch):
        # Resolving must not construct a model or contact anything.
        for backend in ("ollama", "hf"):
            monkeypatch.setattr(cfg, "LLM_BACKEND", backend)
            providers.reset()
            assert callable(providers.get_generator())


class TestRetrieve:
    def test_returns_traceable_contexts(self, wired):
        hits = retrieve("is a burst pipe covered?")
        assert [h["source"] for h in hits] == ["home_insurance_policy.md"] * 2
        assert hits[0]["text"] == "Burst pipe damage is covered."
        assert hits[0]["chunk_id"] == "nfip-sfip-dwelling#aaaaaaaaaaaa"
        assert hits[0]["start"] == 100

    def test_respects_an_explicit_top_k(self, wired):
        assert len(retrieve("q", top_k=1)) == 1

    def test_draws_candidates_before_fusing(self, wired):
        """Fusion needs a deeper candidate list than it returns."""
        from src.rag import CANDIDATE_DEPTH

        retrieve("q")
        assert wired.store.queries[-1]["limit"] == CANDIDATE_DEPTH

    def test_queries_both_retrievers(self, wired):
        retrieve("is a burst pipe covered?")
        assert wired.store.queries, "the dense store was never queried"
        assert wired.bm25.queries, "the lexical index was never queried"

    def test_lexical_query_is_not_prefixed(self, wired):
        """The BGE instruction belongs to the embedder, not to BM25."""
        retrieve("is a burst pipe covered?")
        assert wired.bm25.queries[-1] == "is a burst pipe covered?"

    def test_store_failure_becomes_retrieval_unavailable(self, fake_embedder, make_store):
        providers.set_embedder(fake_embedder)
        providers.set_vector_store(make_store(error=ConnectionRefusedError("refused")))
        with pytest.raises(RetrievalUnavailable):
            retrieve("q")

    def test_original_cause_is_preserved(self, fake_embedder, make_store):
        cause = ConnectionRefusedError("refused")
        providers.set_embedder(fake_embedder)
        providers.set_vector_store(make_store(error=cause))
        with pytest.raises(RetrievalUnavailable) as info:
            retrieve("q")
        assert info.value.__cause__ is cause


def _ctx(text, source, cfr="44 CFR Part 61, Appendix A(1)"):
    return {"text": text, "source": source, "cfr_citation": cfr}


class TestBuildPrompt:
    def test_includes_every_context_and_its_source(self):
        prompt = build_prompt("Q?", [_ctx("clause one", "a.md"), _ctx("clause two", "b.md")])
        assert "a.md" in prompt and "clause one" in prompt
        assert "b.md" in prompt and "clause two" in prompt
        assert prompt.rstrip().endswith("Answer:")

    def test_numbers_the_context_blocks_for_citation(self):
        prompt = build_prompt("Q?", [_ctx("one", "a.md"), _ctx("two", "b.md")])
        assert "[1]" in prompt and "[2]" in prompt
        assert "square brackets" in prompt

    def test_instructs_the_model_to_stay_in_context(self):
        prompt = build_prompt("Q?", [_ctx("c", "s")])
        assert "ONLY" in prompt
        assert "don't know" in prompt

    def test_handles_no_context(self):
        """An empty index still produces a well-formed prompt rather than crashing."""
        prompt = build_prompt("Q?", [])
        assert "Question: Q?" in prompt


class TestAnswer:
    def test_composes_retrieval_and_generation(self, wired):
        result = answer("Is a burst pipe covered?")
        assert result["status"] == STATUS_ANSWERED
        assert result["answer"] == "Yes, a burst pipe is covered."
        assert len(result["citations"]) == 2

    def test_abstains_when_nothing_is_retrieved(self, fake_embedder, make_store, fake_bm25):
        """The one abstention case the dev evidence supports: no evidence at all.

        The generator must not be called - there is nothing to answer from.
        """
        def explode(_):
            raise AssertionError("generator called with no evidence")

        providers.set_embedder(fake_embedder)
        providers.set_vector_store(make_store(hits=[]))
        providers.set_bm25_index(fake_bm25)
        providers.set_generator(explode)

        result = answer("something absent")
        assert result["status"] == "insufficient_evidence"
        assert result["citations"] == []

    def test_generator_failure_propagates_as_generation_unavailable(self, wired):
        providers.set_generator(
            lambda p: (_ for _ in ()).throw(GenerationUnavailable("down"))
        )
        with pytest.raises(GenerationUnavailable):
            answer("q")


class TestCitation:
    def test_carries_full_provenance(self, wired):
        entry = citation(retrieve("q")[0])
        assert set(entry) == {
            "chunk_id", "document_id", "source", "form", "cfr_citation",
            "start", "end", "score", "excerpt",
        }

    def test_excerpt_is_a_prefix_of_the_chunk(self, wired):
        context = retrieve("q")[0]
        entry = citation(context)
        assert context["text"].startswith(entry["excerpt"].removesuffix("..."))

    def test_long_text_is_truncated_with_an_ellipsis(self):
        context = {
            "chunk_id": "d#1", "document_id": "d", "source": "s", "form": "f",
            "cfr_citation": "c", "start": 0, "end": 500, "score": 0.5,
            "text": "x" * 500,
        }
        entry = citation(context, excerpt_chars=100)
        assert entry["excerpt"].endswith("...")
        assert len(entry["excerpt"]) == 103

    def test_excerpt_is_never_generated(self):
        """The excerpt is sliced from the chunk, never produced by a model."""
        context = {
            "chunk_id": "d#1", "document_id": "d", "source": "s", "form": "f",
            "cfr_citation": "c", "start": 10, "end": 30, "score": 0.5,
            "text": "the exact source text",
        }
        assert citation(context)["excerpt"] == "the exact source text"
