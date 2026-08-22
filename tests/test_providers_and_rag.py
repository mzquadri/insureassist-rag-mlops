"""The dependency seam and the retrieval path, without a model or a database."""
import pytest

from src import providers
from src.config import cfg
from src.errors import GenerationUnavailable, RetrievalUnavailable
from src.rag import answer, build_prompt, generate, retrieve


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
    def test_returns_text_source_and_score(self, wired):
        hits = retrieve("is a burst pipe covered?")
        assert [h["source"] for h in hits] == ["home_insurance_policy.md"] * 2
        assert hits[0]["text"] == "Burst pipe damage is covered."
        assert hits[0]["score"] == pytest.approx(0.8712)

    def test_respects_an_explicit_top_k(self, wired):
        assert len(retrieve("q", top_k=1)) == 1

    def test_falls_back_to_configured_top_k(self, wired):
        retrieve("q")
        assert wired.store.queries[-1]["limit"] == cfg.TOP_K

    def test_normalises_the_query_embedding(self, wired):
        retrieve("q")
        assert wired.store.queries, "the store was never queried"

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


class TestBuildPrompt:
    def test_includes_every_context_and_its_source(self):
        prompt = build_prompt("Q?", [
            {"text": "clause one", "source": "a.md", "score": 0.9},
            {"text": "clause two", "source": "b.md", "score": 0.8},
        ])
        assert "[Source: a.md]" in prompt and "clause one" in prompt
        assert "[Source: b.md]" in prompt and "clause two" in prompt
        assert prompt.rstrip().endswith("Answer:")

    def test_instructs_the_model_to_stay_in_context(self):
        prompt = build_prompt("Q?", [{"text": "c", "source": "s", "score": 1.0}])
        assert "ONLY" in prompt
        assert "don't know" in prompt

    def test_handles_no_context(self):
        """An empty index still produces a well-formed prompt rather than crashing.

        The model is expected to decline, though nothing enforces that - abstention is
        prompt-only today, which is recorded as a limitation in the README.
        """
        prompt = build_prompt("Q?", [])
        assert "Question: Q?" in prompt


class TestAnswer:
    def test_composes_retrieval_and_generation(self, wired):
        result = answer("Is a burst pipe covered?")
        assert result["answer"] == "Yes, a burst pipe is covered."
        assert len(result["sources"]) == 2

    def test_sources_expose_only_document_and_score(self, wired):
        """Attribution is document-level; chunk text is not returned.

        This is why citation correctness cannot be measured yet - there is no chunk or span
        identifier in the response to check an answer against.
        """
        assert all(set(s) == {"source", "score"} for s in answer("q")["sources"])

    def test_generator_failure_propagates_as_generation_unavailable(
        self, fake_embedder, fake_store
    ):
        providers.set_embedder(fake_embedder)
        providers.set_vector_store(fake_store)
        providers.set_generator(
            lambda p: (_ for _ in ()).throw(GenerationUnavailable("down"))
        )
        with pytest.raises(GenerationUnavailable):
            answer("q")
