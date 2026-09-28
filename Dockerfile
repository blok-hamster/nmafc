# syntax=docker/dockerfile:1
#
# NMAFC one-container image: exported dashboard + FastAPI on a single port.
#
#   docker compose up                    # dash + API on :8000, LanceDB in ./data
#   docker compose --profile pgvector up # adds a pgvector Cold ROM service
#
# LanceDB is embedded; the only mounted state is the /app/data volume. For a
# remote Cold ROM, set NMAFC_COLD_URI to a postgres:// URL and add
# psycopg2-binary (nmafc[postgres,web]) or run the compose profile.

# ── Stage 1: export the Next.js dashboard (web-ui/out) ──
FROM node:22-alpine AS ui
WORKDIR /web-ui
COPY web-ui/package.json web-ui/package-lock.json ./
RUN npm ci
COPY web-ui/ ./
# NMAFC_EXPORT=1 switches next.config.ts to output:'export' (no rewrites,
# since FastAPI serves the same origin — see next.config.ts).
RUN NMAFC_EXPORT=1 npm run build

# ── Stage 2: Python runtime ──
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NMAFC_STATIC_UI_DIR=/app/web-ui/out \
    NMAFC_CONFIG_PATH=/app/configs/default.toml

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN pip install --no-cache-dir '.[web]'

COPY --from=ui /web-ui/out ./web-ui/out

EXPOSE 8000

VOLUME /app/data

HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

CMD ["python", "-m", "uvicorn", "nmafc.web.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]