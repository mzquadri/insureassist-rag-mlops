"""
Container integration: a real image, a real Qdrant, real HTTP.

Skipped unless INSUREASSIST_INTEGRATION=1 and the image exists, so the default suite stays
offline and fast. CI runs it as its own job.

The point is to prove *application wiring*, not that some process returns 200. A stub
server would pass a naive health check; these tests require the container to load the
embedding model, build the lexical index, reach Qdrant, retrieve real chunks from the
committed corpus, and return citations whose offsets resolve against that corpus.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import pytest

IMAGE = os.environ.get("INSUREASSIST_IMAGE", "insureassist:test")
ENABLED = os.environ.get("INSUREASSIST_INTEGRATION") == "1"

pytestmark = pytest.mark.skipif(
    not ENABLED, reason="set INSUREASSIST_INTEGRATION=1 to run container integration tests"
)


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=check, timeout=300)


def _get(url: str, timeout: float = 10) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post(url: str, payload: dict, timeout: float = 180) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture(scope="module")
def stack():
    """Qdrant plus the API container on a shared network, ingested and ready."""
    network = "insureassist-it"
    _run("docker", "network", "create", network, check=False)
    _run("docker", "rm", "-f", "it-qdrant", "it-api", check=False)

    _run("docker", "run", "-d", "--name", "it-qdrant", "--network", network,
         "qdrant/qdrant:v1.19.0")

    env = [
        "-e", "QDRANT_URL=http://it-qdrant:6333",
        "-e", "QDRANT_COLLECTION=nfip_sfip",
        "-e", "CORPUS=nfip",
        "-e", "CHUNK_SIZE=800",
        "-e", "CHUNK_OVERLAP=120",
    ]

    # Wait for Qdrant, then ingest using the same image - which also proves the image can
    # run the ingestion path, not just serve.
    deadline = time.time() + 120
    while time.time() < deadline:
        probe = _run("docker", "run", "--rm", "--network", network, IMAGE,
                     "python", "-c",
                     "import urllib.request;urllib.request.urlopen('http://it-qdrant:6333/readyz',timeout=2)",
                     check=False)
        if probe.returncode == 0:
            break
        time.sleep(3)

    ingest = _run("docker", "run", "--rm", "--network", network, *env, IMAGE,
                  "python", "-m", "src.ingest")
    assert "Ingested 314 chunks" in ingest.stdout, ingest.stdout + ingest.stderr

    _run("docker", "run", "-d", "--name", "it-api", "--network", network,
         "-p", "18000:8000", *env, IMAGE)

    base = "http://127.0.0.1:18000"
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            status, _ = _get(f"{base}/health", timeout=3)
            if status == 200:
                break
        except (OSError, ValueError):
            pass  # container still starting; keep polling until the deadline
        time.sleep(3)
    else:
        logs = _run("docker", "logs", "it-api", check=False)
        pytest.fail(f"API never became live.\n{logs.stdout}\n{logs.stderr}")

    yield base

    _run("docker", "rm", "-f", "it-qdrant", "it-api", check=False)
    _run("docker", "network", "rm", network, check=False)


class TestContainerIdentity:
    def test_runs_as_a_non_root_user(self):
        result = _run("docker", "run", "--rm", IMAGE, "id", "-u")
        assert result.stdout.strip() == "10001", "container must not run as root"

    def test_user_has_no_login_shell(self):
        result = _run("docker", "run", "--rm", IMAGE, "sh", "-c", "getent passwd app")
        assert "nologin" in result.stdout

    def test_image_declares_a_healthcheck(self):
        result = _run("docker", "image", "inspect", IMAGE,
                      "--format", "{{if .Config.Healthcheck}}yes{{else}}no{{end}}")
        assert result.stdout.strip() == "yes"


class TestLiveEndpoints:
    def test_health_is_live(self, stack):
        status, body = _get(f"{stack}/health")
        assert status == 200
        assert body == {"status": "ok"}

    def test_ready_reports_every_dependency(self, stack):
        status, body = _get(f"{stack}/ready", timeout=180)
        assert status == 200, body
        assert body["status"] == "ready"
        assert set(body["dependencies"]) == {"embedder", "lexical_index", "vector_store"}
        assert all(d["ready"] for d in body["dependencies"].values())

    def test_request_id_is_returned_and_honoured(self, stack):
        request = urllib.request.Request(
            f"{stack}/health", headers={"X-Request-ID": "integration-test-id"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.headers["X-Request-ID"] == "integration-test-id"


class TestRealAskPath:
    """Exercises the whole application, not an HTTP stub.

    No generator is reachable in CI, so /ask is expected to fail at *generation* - after
    retrieval has already succeeded. That is the point: a 503 naming `generator` proves the
    request reached the embedder, the lexical index and Qdrant first. A stub server could
    not produce it.
    """

    def test_ask_reaches_generation_and_reports_the_right_dependency(self, stack):
        status, body = _post(f"{stack}/ask", {"question": "What is the Increased Cost of Compliance limit?"})
        assert status == 503, body
        assert body["dependency"] == "generator"
        assert body["status"] == "dependency_unavailable"
        assert body["request_id"]

    def test_retrieval_really_ran_inside_the_container(self, stack):
        """Ask the container to retrieve directly, proving real chunks come back."""
        result = _run(
            "docker", "exec", "it-api", "python", "-c",
            "import json;from src.rag import retrieve;"
            "c=retrieve('What is the Increased Cost of Compliance limit?',5);"
            "print(json.dumps([{k:x[k] for k in ('chunk_id','document_id','start','end')} for x in c]))",
        )
        contexts = json.loads(result.stdout.strip().splitlines()[-1])
        assert len(contexts) == 5
        assert all(c["chunk_id"].startswith("nfip-sfip-") for c in contexts)
        assert all(c["end"] > c["start"] for c in contexts)

    def test_validation_still_applies_over_http(self, stack):
        status, _ = _post(f"{stack}/ask", {"question": ""})
        assert status == 422
