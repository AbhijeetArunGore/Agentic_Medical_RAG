# ==============================================================================
# Multi-Stage Hardened Dockerfile for Agentic Medical RAG Microservice
# ==============================================================================

# ----------------- STAGE 1: Builder -----------------
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /install

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ----------------- STAGE 2: Runtime -----------------
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/install/bin:$PATH" \
    PYTHONPATH="/app" \
    ENVIRONMENT="production" \
    VECTOR_DB_PATH="/app/vector_db"

WORKDIR /app

# Install minimal curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python dependencies from builder stage
COPY --from=builder /install /usr/local

# Create non-root user and persistent directories
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser && \
    mkdir -p /app/vector_db /app/data /app/docs && \
    chown -R appuser:appgroup /app

# Copy application source code
COPY --chown=appuser:appgroup . /app

# Switch to non-root runtime user
USER appuser

# Healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000 8501

CMD ["uvicorn", "app.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
