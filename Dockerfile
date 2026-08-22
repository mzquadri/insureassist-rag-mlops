# Container image for the RAG API.
#
# Two stages, and this time genuinely so: dependencies are installed into a virtualenv in
# the builder and only the built environment is copied forward, which keeps pip, its cache
# and the build toolchain out of the runtime image.
#
# Known boundary, stated rather than hidden: the embedding model is not baked in. It
# downloads on first use into HF_HOME, so a cold container needs network access and its
# first request is slow. Pre-baking it would roughly double the image; for a reference
# project the trade is not obviously worth it, and the Kubernetes manifests mount a cache
# volume to make the download survive restarts.

FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/home/app/.cache/huggingface

# Non-root. A fixed uid/gid so a mounted cache volume has predictable ownership.
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app src/ ./src/
COPY --chown=app:app data/ ./data/
COPY --chown=app:app eval/retrieval_config.json ./eval/retrieval_config.json

RUN mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app/.cache
USER app

EXPOSE 8000

# Liveness only, matching the endpoint's contract. Readiness is /ready and is checked by
# the orchestrator, not by Docker - a container whose dependencies are down is still alive.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2).status==200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
