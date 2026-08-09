# Phase 4 — container image for the RAG API.
# Multi-stage keeps the final image smaller (a production best practice).

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

EXPOSE 8000
# Health endpoint is /health (used by Kubernetes probes)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
