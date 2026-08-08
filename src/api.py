"""
Phase 1 — FastAPI service.

Run:
    uvicorn src.api:app --reload
Then open http://localhost:8000/docs  and try the /ask endpoint.
"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.rag import answer

app = FastAPI(title="InsureAssist RAG API", version="0.1.0")


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    source: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    """Kubernetes/GKE will use this later for liveness/readiness checks."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    return answer(req.question)
