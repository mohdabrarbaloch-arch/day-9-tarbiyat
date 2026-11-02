FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build tools first, in their own layer, so a code change does not reinstall them.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY ARCHITECTURE.md README.md ./

# Run unprivileged. A model server has no business being root.
RUN useradd --create-home --uid 10001 tarbiyat \
    && mkdir -p /app/data \
    && chown -R tarbiyat:tarbiyat /app
USER tarbiyat

ENV DATABASE_URL=sqlite:////app/data/tarbiyat.db \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
