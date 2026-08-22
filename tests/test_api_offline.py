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
    def test_returns_answer_and_sources(self, client, wired):
        response = client.post("/ask", json={"question": "Is a burst pipe covered?"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Yes, a burst pipe is covered."
        assert body["sources"] == [
            {"source": "home_insurance_policy.md", "score": 0.871},
            {"source": "home_insurance_policy.md", "score": 0.703},
        ]

    def test_scores_are_rounded_to_three_places(self, client, wired):
        scores = [s["score"] for s in client.post(
            "/ask", json={"question": "q"}).json()["sources"]]
        assert scores == [0.871, 0.703]  # from 0.8712 / 0.7031

    def test_the_question_reaches_the_embedder(self, client, wired):
        """The question is embedded, carrying the BGE query instruction prefix.

        BGE is asymmetric - queries take an instruction and passages do not - so what
        reaches the embedder is the prefixed question, not the raw one.
        """
        from src.config import cfg

        client.post("/ask", json={"question": "Does it cover hail?"})
        assert wired.embedder.calls == [cfg.BGE_QUERY_PREFIX + "Does it cover hail?"]

    def test_the_prompt_carries_the_retrieved_context(self, client, fake_embedder, fake_store):
        seen = {}

        def capture(prompt):
            seen["prompt"] = prompt
            return "answer"

        providers.set_embedder(fake_embedder)
        providers.set_vector_store(fake_store)
        providers.set_generator(capture)

        client.post("/ask", json={"question": "Is a burst pipe covered?"})
        assert "Burst pipe damage is covered." in seen["prompt"]
        assert "[Source: home_insurance_policy.md]" in seen["prompt"]
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

    def test_qdrant_unavailable_maps_to_503(self, client, fake_embedder, make_store):
        providers.set_embedder(fake_embedder)
        providers.set_vector_store(make_store(error=ConnectionRefusedError("connection refused")))
        providers.set_generator(lambda p: "unused")

        response = client.post("/ask", json={"question": "anything"})
        assert response.status_code == 503
        assert response.json()["dependency"] == "retrieval"

    def test_embedding_model_unavailable_maps_to_503(self, client, fake_store):
        def cannot_load(*_args, **_kwargs):
            raise RetrievalUnavailable("could not load embedding model")

        class BrokenEmbedder:
            encode = staticmethod(cannot_load)

        providers.set_embedder(BrokenEmbedder())
        providers.set_vector_store(fake_store)
        providers.set_generator(lambda p: "unused")

        response = client.post("/ask", json={"question": "anything"})
        assert response.status_code == 503
        assert response.json()["dependency"] == "retrieval"

    def test_generator_unavailable_maps_to_503(self, client, fake_embedder, fake_store):
        def down(_prompt):
            raise GenerationUnavailable("ollama did not answer")

        providers.set_embedder(fake_embedder)
        providers.set_vector_store(fake_store)
        providers.set_generator(down)

        response = client.post("/ask", json={"question": "anything"})
        assert response.status_code == 503
        assert response.json()["dependency"] == "generator"

    def test_failure_body_explains_which_dependency(self, client, fake_embedder, fake_store):
        providers.set_embedder(fake_embedder)
        providers.set_vector_store(fake_store)
        providers.set_generator(
            lambda p: (_ for _ in ()).throw(GenerationUnavailable("ollama refused"))
        )
        body = client.post("/ask", json={"question": "q"}).json()
        assert set(body) == {"detail", "dependency"}
        assert "ollama refused" in body["detail"]

    def test_health_still_answers_while_dependencies_are_down(
        self, client, fake_embedder, make_store
    ):
        """Liveness must not fail just because a dependency is unreachable.

        Otherwise Kubernetes would restart a perfectly healthy pod during a Qdrant outage.
        """
        providers.set_embedder(fake_embedder)
        providers.set_vector_store(make_store(error=ConnectionRefusedError("down")))
        assert client.post("/ask", json={"question": "q"}).status_code == 503
        assert client.get("/health").status_code == 200
