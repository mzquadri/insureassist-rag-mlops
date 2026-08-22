"""
FastAPI service.

Run:
    uvicorn src.api:app --reload
Then open http://localhost:8000/docs and try the /ask endpoint.

Importing this module does not load a model, open a socket, or require Ollama. Everything
heavy is built on first request via `src.providers`.
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.errors import DependencyUnavailable
from src.rag import answer

app = FastAPI(title="InsureAssist RAG API", version="0.1.0")


class AskRequest(BaseModel):
    # A blank question retrieves nothing useful and still costs a generator call, so it is
    # rejected at the edge rather than answered badly.
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    source: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.exception_handler(DependencyUnavailable)
async def _dependency_unavailable(request: Request, exc: DependencyUnavailable) -> JSONResponse:
    """
    A missing backing service is a 503, not a 500.

    The request was well-formed and would likely succeed later, so the caller is told to
    retry rather than to fix its input. `dependency` says which half of the pipeline is
    down without requiring anyone to parse the message.
    """
    return JSONResponse(
        status_code=503,
        content={"detail": str(exc), "dependency": exc.dependency},
    )


@app.get("/health")
def health():
    """
    Liveness only: is this process running?

    It deliberately reports nothing about Qdrant, the embedding model, or the generator,
    so it must not be used as a readiness signal - a process can pass this check and still
    be unable to answer. A real readiness endpoint is a later phase; until then the
    Kubernetes manifests in `k8s/` use this for both probes, which is documented there as
    a known limitation.
    """
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    return answer(req.question)
