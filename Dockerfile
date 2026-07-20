FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# uv is required only to create the virtual environment; it is not shipped in
# the runtime image. No compiler toolchain or curl is needed by this project.
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project


FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # mem0 writes its telemetry configuration at import time. The non-login
    # application user has /nonexistent as its home, so direct that state to
    # a writable ephemeral directory instead.
    MEM0_DIR=/tmp/.mem0 \
    PATH="/app/.venv/bin:${PATH}"

WORKDIR /app

# Keep the runtime account's identity stable so a deployment can safely grant
# it write access to host-mounted persistent data.
ARG APP_UID=10001
ARG APP_GID=10001
RUN addgroup --system --gid "$APP_GID" app \
    && adduser --system --uid "$APP_UID" --ingroup app app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app customer_support_agent ./customer_support_agent
COPY --chown=app:app main.py app.py ./

RUN mkdir -p data/chroma_rag data/chroma_mem0 knowledge_base \
    && chown -R app:app /app

USER app

EXPOSE 8000 8501

CMD ["python", "main.py"]
