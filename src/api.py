"""
FastAPI service.

Run:
    uvicorn src.api:app --reload

Importing this module does not load a model, open a socket, or require Ollama. Everything
heavy is built on first use via `src.providers`.

Three endpoints with deliberately different contracts:

  GET  /health  liveness  - is this process running? Never touches a dependency.
  GET  /ready   readiness - can it actually serve? Checks every serving dependency.
  POST /ask     the work.

The split matters. `/health` used to serve both roles, which meant Kubernetes marked a pod
ready while Qdrant was unreachable and routed traffic to something that could only fail.
"""
from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import cfg
from src.errors import DependencyUnavailable
from src.rag import answer

logger = logging.getLogger("insureassist")

app = FastAPI(title="InsureAssist RAG API", version="0.2.0")

REQUEST_ID_HEADER = "X-Request-ID"


class AskRequest(BaseModel):
    # A blank question retrieves nothing useful and still costs a generator call, so it is
    # rejected at the edge rather than answered badly.
    question: str = Field(min_length=1, max_length=2000)


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    form: str
    cfr_citation: str
    start: int
    end: int
    score: float
    excerpt: str


class AskResponse(BaseModel):
    status: str
    answer: str
    citations: list[Citation]
    request_id: str


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a request ID to every response, honouring one supplied by the caller.

    Makes a single request traceable through the logs without correlating on timestamps.
    """
    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers[REQUEST_ID_HEADER] = request_id
    return response


@app.exception_handler(DependencyUnavailable)
async def _dependency_unavailable(request: Request, exc: DependencyUnavailable) -> JSONResponse:
    """A missing backing service is a 503, not a 500.

    The request was well-formed and would likely succeed later, so the caller is told to
    retry rather than to fix its input. The message is the exception's own text, which is
    written to name the dependency and never interpolates a credential or a payload.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning("dependency unavailable: %s (request_id=%s)", exc.dependency, request_id)
    return JSONResponse(
        status_code=503,
        content={
            "status": "dependency_unavailable",
            "detail": str(exc),
            "dependency": exc.dependency,
            "request_id": request_id,
        },
        headers={REQUEST_ID_HEADER: request_id},
    )


@app.exception_handler(Exception)
async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
    """Never leak an internal exception to a caller.

    The detail is logged; the response carries a request ID so a report can be traced back
    to the log line without the caller ever seeing a stack trace or an internal path.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("unhandled error (request_id=%s)", request_id)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": "internal error",
            "request_id": request_id,
        },
        headers={REQUEST_ID_HEADER: request_id},
    )


@app.get("/health")
def health():
    """Liveness only: is this process running?

    Reports nothing about Qdrant, the embedding model or the generator, so it must never be
    used as a readiness signal. Use /ready for that.
    """
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request):
    """Readiness: can this process actually serve a request?

    Checks each serving dependency and reports them individually, so a 503 says which one
    is missing instead of merely that something is. Returns 503 unless all are ready -
    a pod that cannot retrieve should not receive traffic.
    """
    from src.providers import get_bm25_index, get_embedder, get_vector_store

    dependencies: dict[str, dict] = {}

    def check(name: str, probe):
        try:
            probe()
            dependencies[name] = {"ready": True}
        except Exception as exc:  # noqa: BLE001 - readiness must never raise
            dependencies[name] = {"ready": False, "detail": type(exc).__name__}

    check("embedder", get_embedder)
    check("lexical_index", get_bm25_index)
    check(
        "vector_store",
        lambda: get_vector_store().get_collection(cfg.QDRANT_COLLECTION),
    )

    ready_now = all(d["ready"] for d in dependencies.values())
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=200 if ready_now else 503,
        content={
            "status": "ready" if ready_now else "not_ready",
            "dependencies": dependencies,
            "request_id": request_id,
        },
        headers={REQUEST_ID_HEADER: request_id},
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request):
    result = answer(req.question)
    return {**result, "request_id": getattr(request.state, "request_id", "unknown")}
