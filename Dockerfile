# Container image for the RAG API.
#
# Single stage. An earlier comment here described this as multi-stage; it never was.
#
# Known gaps, tracked rather than hidden: the image runs as root, declares no
# HEALTHCHECK, and installs unpinned dependencies, so two builds can differ. The
# embedding model is not baked in either - it downloads on first request, so a cold
# container needs network access and is slow to answer.

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
