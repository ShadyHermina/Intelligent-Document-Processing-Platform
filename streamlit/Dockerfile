# ── Base image ────────────────────────────────────────────────────
# FROM sets the starting point for this image.
# python:3.11-slim is a minimal Debian-based image with Python 3.11
# pre-installed. "slim" strips out compilers, documentation, and other
# tools that are needed to build packages but not to run them.
#
# Alternative considered: python:3.11-alpine
# Alpine is even smaller, but it uses musl libc instead of glibc.
# Several Python packages (including some numpy/pandas dependencies)
# assume glibc and fail to install on Alpine without extra work.
# slim gives us small size without that compatibility risk.
FROM python:3.11-slim

# curl: needed for Docker Compose health checks and inter-container
# debugging. Same single-RUN chaining pattern as the FastAPI Dockerfile.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────
# WORKDIR creates this directory inside the container and sets it as
# the current directory for all subsequent instructions.
# /app is the convention. There is nothing special about the name —
# it is just universally understood by engineers reading the file.
WORKDIR /app

# ── Install dependencies ──────────────────────────────────────────
# We copy requirements.txt BEFORE copying the rest of the source code.
# This is the most important layer-caching optimization in any Dockerfile.
#
# Why: Docker rebuilds a layer only if its inputs changed.
# If we copied all source files first, then requirements.txt, any change
# to app.py would invalidate the pip install layer and force a full
# re-download of all packages on every build.
# By copying requirements.txt alone first, the pip install layer is
# only invalidated when requirements.txt itself changes.
COPY requirements.txt .

# RUN executes a shell command during the build.
# --no-cache-dir tells pip not to store the downloaded package files
# after installation. We don't need them again inside the image,
# and they would just add size.
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────
# Now we copy the rest of the files. Because this comes AFTER the
# pip install layer, changing app.py does not re-trigger pip install.
COPY . .

# ── Expose ───────────────────────────────────────────────────────
# EXPOSE documents that the process inside this container listens
# on port 8501. It does not actually open the port — that is done
# in docker-compose.yml. Think of EXPOSE as a label for humans and
# tooling, not a firewall rule.
# 8501 is Streamlit's default port.
EXPOSE 8501

# ── Entry point ───────────────────────────────────────────────────
# CMD is the command Docker runs when the container starts.
# We use the JSON array form (["cmd", "arg1", "arg2"]) rather than
# the shell string form ("cmd arg1 arg2") because:
# - The JSON form runs the command directly (PID 1 = streamlit)
# - The shell string form wraps it in /bin/sh -c, meaning sh becomes
#   PID 1 and streamlit is a child process. This breaks Docker's
#   ability to forward shutdown signals (SIGTERM) to the actual app,
#   causing slow or unclean container stops.
#
# --server.address=0.0.0.0
#   By default Streamlit binds to 127.0.0.1 (localhost inside the
#   container). That makes it unreachable from outside the container.
#   0.0.0.0 means "listen on all network interfaces" — required for
#   Docker networking to work.
#
# --server.port=8501
#   Explicit is better than implicit. We declare the port rather than
#   relying on the default so it is obvious and grep-able.
#
# --server.headless=true
#   Disables the "Please wait..." browser-open attempt that Streamlit
#   makes on startup. Inside a container there is no browser to open.
#   Without this flag you get a harmless but noisy warning on every start.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]