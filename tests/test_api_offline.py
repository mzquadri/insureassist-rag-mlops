"""
The API layer, exercised entirely offline.

The first test is the load-bearing one: if importing the app ever loads a model or opens a
socket again, every other test here becomes slow, flaky, and network-dependent.
"""
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from src import providers
from src.api import app
from src.errors import GenerationUnavailable, RetrievalUnavailable


@pytest.fixture
def client():
    return TestClient(app)


class TestImportIsInert:
    def test_importing_the_app_loads_no_heavy_modules(self):
        """Importing the API must not drag in torch, sentence_transformers, or a client.

        Asserted on `sys.modules` rather than by timing, so the test states the actual
        requirement instead of guessing at a duration.
        """
        import src.api  # noqa: F401  (already imported; this documents the subject)

        for heavy in ("sentence_transformers", "torch", "qdrant_client"):
            assert heavy not in sys.modules, (
                f"importing src.api pulled in {heavy!r}; the provider seam has regressed"
            )

    def test_importing_the_app_with_networking_disabled(self):
        """Import the app in a subprocess where every socket operation raises.

        This is the guarantee a CI runner and a cold container actually depend on, and it
        cannot be faked by module bookkeeping - so it runs for real, in isolation.
        """
        # Block the operations that reach the network, but leave `socket.socket` itself
        # intact: the standard library subclasses it (`class SSLSocket(socket)`), so
        # replacing the class outright breaks importing ssl and proves nothing.
        code = (
            "import socket, sys\n"
            "def _blocked(*a, **k):\n"
            "    raise OSError('network disabled for this test')\n"
            "socket.socket.connect = _blocked\n"
            "socket.socket.connect_ex = _blocked\n"
            "socket.create_connection = _blocked\n"
            "socket.getaddrinfo = _blocked\n"
            "import src.api\n"
            "assert 'sentence_transformers' not in sys.modules\n"
            "assert 'qdrant_client' not in sys.modules\n"
            "assert src.api.app.title == 'InsureAssist RAG API'\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120, check=False,
        )
        assert result.returncode == 0, (
            f"importing src.api needed the network.\nstdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert "OK" in result.stdout


class TestHealth:
    def test_health_reports_liveness(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_does_not_touch_dependencies(self, client):
        """`/health` answers with no providers configured at all.

        That is precisely why it must not be used as a readiness probe: it cannot fail for
        a dependency reason. The Kubernetes manifests currently use it for readiness, which
        is documented as a known gap in `k8s/README.md`.
        """
        providers.reset()
        assert client.get("/health").status_code == 200


class TestAskHappyPath:
    def test_returns_answer_status_and_citations(self, client, wired):
        response = client.post("/ask", json={"question": "Is a burst pipe covered?"})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "answered"
        assert body["answer"] == "Yes, a burst pipe is covered."
        assert len(body["citations"]) == 2
        assert body["request_id"]

    def test_citations_are_traceable(self, client, wired):
        """Every citation carries what is needed to find the text again."""
        citations = client.post("/ask", json={"question": "q"}).json()["citations"]
        for entry in citations:
            assert set(entry) == {
                "chunk_id", "document_id", "source", "form", "cfr_citation",
                "start", "end", "score", "excerpt",
            }
            assert entry["end"] > entry["start"]
            assert entry["chunk_id"].startswith(entry["document_id"] + "#")

    def test_excerpt_is_quoted_not_generated(self, client, wired):
        citations = client.post("/ask", json={"question": "q"}).json()["citations"]
        assert citations[0]["excerpt"].startswith("Burst pipe damage is covered.")

    def test_the_question_reaches_the_embedder(self, client, wired):
        """The question is embedded, carrying the BGE query instruction prefix.

        BGE is asymmetric - queries take an instruction and passages do not - so what
        reaches the embedder is the prefixed question, not the raw one.
        """
        from src.config import cfg

        client.post("/ask", json={"question": "Does it cover hail?"})
        assert wired.embedder.calls == [cfg.BGE_QUERY_PREFIX + "Does it cover hail?"]

    def test_the_prompt_carries_the_retrieved_context(self, client, wired):
        seen = {}

        def capture(prompt):
            seen["prompt"] = prompt
            return "answer"

        providers.set_generator(capture)
        client.post("/ask", json={"question": "Is a burst pipe covered?"})
        assert "Burst pipe damage is covered." in seen["prompt"]
        assert "home_insurance_policy.md" in seen["prompt"]
        assert "Is a burst pipe covered?" in seen["prompt"]


class TestAskValidation:
    def test_missing_question_is_rejected(self, client):
        assert client.post("/ask", json={}).status_code == 422

    def test_empty_question_is_rejected(self, client):
        assert client.post("/ask", json={"question": ""}).status_code == 422

    def test_wrong_type_is_rejected(self, client):
        assert client.post("/ask", json={"question": 42}).status_code == 422

    def test_oversized_question_is_rejected(self, client):
        assert client.post("/ask", json={"question": "x" * 5000}).status_code == 422

    def test_validation_happens_before_any_dependency_is_touched(self, client):
        """A malformed request must not reach the embedder or the generator."""
        def explode(_):
            raise AssertionError("generator was called for an invalid request")

        providers.set_generator(explode)
        assert client.post("/ask", json={"question": ""}).status_code == 422


class TestDependencyFailures:
    """A missing backing service is a 503 with a machine-readable cause, not a 500."""

    def test_qdrant_unavailable_maps_to_503(self, client, fake_embedder, make_store, fake_bm25):
        providers.set_embedder(fake_embedder)
        providers.set_vector_store(make_store(error=ConnectionRefusedError("refused")))
        providers.set_bm25_index(fake_bm25)
        providers.set_generator(lambda p: "unused")

        response = client.post("/ask", json={"question": "anything"})
        assert response.status_code == 503
        assert response.json()["dependency"] == "retrieval"

    def test_embedding_model_unavailable_maps_to_503(self, client, fake_store, fake_bm25):
        def cannot_load(*_args, **_kwargs):
            raise RetrievalUnavailable("could not load embedding model")

        class BrokenEmbedder:
            encode = staticmethod(cannot_load)

        providers.set_embedder(BrokenEmbedder())
        providers.set_vector_store(fake_store)
        providers.set_bm25_index(fake_bm25)
        providers.set_generator(lambda p: "unused")

        response = client.post("/ask", json={"question": "anything"})
        assert response.status_code == 503
        assert response.json()["dependency"] == "retrieval"

    def test_generator_unavailable_maps_to_503(self, client, wired):
        def down(_prompt):
            raise GenerationUnavailable("ollama did not answer")

        providers.set_generator(down)

        response = client.post("/ask", json={"question": "anything"})
        assert response.status_code == 503
        assert response.json()["dependency"] == "generator"

    def test_failure_body_explains_which_dependency(self, client, wired):
        providers.set_generator(
            lambda p: (_ for _ in ()).throw(GenerationUnavailable("ollama refused"))
        )
        body = client.post("/ask", json={"question": "q"}).json()
        assert set(body) == {"status", "detail", "dependency", "request_id"}
        assert "ollama refused" in body["detail"]

    def test_health_still_answers_while_dependencies_are_down(
        self, client, fake_embedder, make_store, fake_bm25
    ):
        """Liveness must not fail just because a dependency is unreachable.

        Otherwise Kubernetes would restart a perfectly healthy pod during a Qdrant outage.
        """
        providers.set_embedder(fake_embedder)
        providers.set_vector_store(make_store(error=ConnectionRefusedError("down")))
        providers.set_bm25_index(fake_bm25)
        assert client.post("/ask", json={"question": "q"}).status_code == 503
        assert client.get("/health").status_code == 200


class TestReadiness:
    """`/ready` must reflect dependencies; `/health` must not."""

    def test_ready_when_every_dependency_resolves(self, client, wired):
        class Store:
            def get_collection(self, name):
                return object()

            def query_points(self, **kwargs):
                return wired.store.query_points(**kwargs)

        providers.set_vector_store(Store())
        response = client.get("/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert set(body["dependencies"]) == {"embedder", "lexical_index", "vector_store"}

    def test_not_ready_when_the_vector_store_is_down(self, client, fake_embedder, fake_bm25):
        class Broken:
            def get_collection(self, name):
                raise ConnectionRefusedError("qdrant down")

        providers.set_embedder(fake_embedder)
        providers.set_bm25_index(fake_bm25)
        providers.set_vector_store(Broken())

        response = client.get("/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "not_ready"
        assert body["dependencies"]["vector_store"]["ready"] is False
        assert body["dependencies"]["embedder"]["ready"] is True

    def test_health_stays_200_when_ready_is_503(self, client, fake_embedder, fake_bm25):
        class Broken:
            def get_collection(self, name):
                raise ConnectionRefusedError("down")

        providers.set_embedder(fake_embedder)
        providers.set_bm25_index(fake_bm25)
        providers.set_vector_store(Broken())

        assert client.get("/ready").status_code == 503
        assert client.get("/health").status_code == 200


class TestRequestId:
    def test_generated_when_absent(self, client):
        assert client.get("/health").headers["X-Request-ID"]

    def test_caller_supplied_id_is_honoured(self, client):
        response = client.get("/health", headers={"X-Request-ID": "abc-123"})
        assert response.headers["X-Request-ID"] == "abc-123"

    def test_id_appears_in_a_failure_body(self, client, wired):
        providers.set_generator(
            lambda p: (_ for _ in ()).throw(GenerationUnavailable("down"))
        )
        body = client.post("/ask", json={"question": "q"},
                           headers={"X-Request-ID": "trace-me"}).json()
        assert body["request_id"] == "trace-me"
