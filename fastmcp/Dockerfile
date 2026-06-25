# ── Base image ────────────────────────────────────────────────────
# python:3.11-slim — same reasoning as all other containers.
# Consistent base images across all services means consistent
# behavior and easier security patching: one base image to update.
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────
# curl: needed for health checks from Docker Compose and for
# manual inter-container connectivity testing during debugging.
# Same single-RUN chaining pattern as the FastAPI Dockerfile —
# one layer keeps the apt cache deletion effective.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────
# requirements.txt copied first, source code copied second.
# This is the layer-caching optimization: pip install only reruns
# when requirements.txt changes, not when server.py changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────
COPY . .

# ── Expose ───────────────────────────────────────────────────────
# FastMCP's Streamable HTTP transport listens on 8080 by default.
# We keep the default rather than changing it — no reason to
# introduce a non-standard port when the default is unambiguous.
# 8080 is the conventional "alternative HTTP" port, universally
# understood as "HTTP, but not the primary one."
EXPOSE 8080

# ── Entry point ───────────────────────────────────────────────────
# We run server.py directly with Python rather than using a separate
# ASGI server command. FastMCP 2.x's mcp.run() call is self-contained:
# it starts its own internal HTTP server when transport="streamable-http".
# There is no uvicorn command to invoke — FastMCP handles it internally.
CMD ["python", "server.py"]