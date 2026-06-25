# ── Base image ────────────────────────────────────────────────────
# Same reasoning as the Streamlit image: python:3.11-slim gives us
# a glibc-compatible minimal Python environment.
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────
# We need curl inside this container for two reasons:
# 1. Docker Compose health checks use curl to probe /health
# 2. Manual debugging: you can exec into the container and curl
#    other services to test inter-container connectivity
#
# apt-get update refreshes the package index.
# apt-get install -y installs without prompting for confirmation.
# --no-install-recommends prevents apt from pulling in suggested
# packages we didn't ask for — keeps the image smaller.
# rm -rf /var/lib/apt/lists/* deletes the package index cache after
# installation. We don't need it at runtime and it adds ~30MB.
# All four commands are chained with && in a single RUN instruction.
# Why: each RUN instruction creates a new image layer. If we split
# these into four separate RUN instructions, the apt cache deletion
# would be in a different layer from the installation — Docker's
# layer system means the cache files would still exist in the
# installation layer even after deletion. One RUN = one layer = 
# the deletion actually reduces image size.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────
# Copy requirements first to exploit Docker layer caching.
# pip install runs only when requirements.txt changes, not on every
# code change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────
COPY . .

# ── Expose ───────────────────────────────────────────────────────
# FastAPI/uvicorn will listen on 8000 inside the container.
# 8000 is the uvicorn default and the Python web server convention.
EXPOSE 8000

# ── Entry point ───────────────────────────────────────────────────
# uvicorn is the ASGI server that runs our FastAPI app.
#
# main:app means: in the file main.py, find the object named app.
# That object is the FastAPI() instance we create in main.py.
#
# --host 0.0.0.0
#   Same reasoning as Streamlit: bind to all interfaces so Docker
#   networking can reach the process from outside the container.
#
# --port 8000
#   Explicit port declaration. Matches the EXPOSE above.
#
# --log-level info
#   Uvicorn's access log verbosity. "info" logs every request.
#   In production you might use "warning" to reduce noise, but
#   during Phase 0 we want to see every request hit the container
#   so we can confirm load balancing is working.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]