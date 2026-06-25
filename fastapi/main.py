# main.py — FastAPI backend stub
#
# Phase 0 purpose: start cleanly, respond to /health,
# and identify which instance is responding.
# No business logic. No database calls. No auth.

import os
from fastapi import FastAPI
from pydantic import BaseModel

# FastAPI() creates the ASGI application object.
# The title and version appear in the auto-generated OpenAPI docs
# at /docs — useful for debugging later phases.
app = FastAPI(
    title="Intelligent Document Processing Platform — API",
    version="0.1.0"
)

# os.getenv reads environment variables injected by Docker Compose.
# INSTANCE_ID will be set to "fastapi_a" or "fastapi_b" in
# docker-compose.yml so we can tell the two instances apart in logs.
# LOG_LEVEL is read here for reference — uvicorn actually reads it
# from its own CLI flag, but having it accessible in Python lets us
# use it for application-level logging configuration in later phases.
INSTANCE_ID = os.getenv("INSTANCE_ID", "fastapi_unknown")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")


# ── Response model ────────────────────────────────────────────────
# Pydantic BaseModel defines the shape of the JSON response.
# FastAPI uses this to:
# 1. Serialize the return value to JSON automatically
# 2. Generate the correct schema in /docs
# 3. Validate the response structure at runtime
#
# Alternative: return a plain dict — {"status": "ok"}.
# That works, but gives up type safety and OpenAPI documentation.
# We use a model even here so the pattern is established from day one.
class HealthResponse(BaseModel):
    status: str
    instance: str


# ── Health endpoint ───────────────────────────────────────────────
# @app.get registers a GET handler at the path /health.
# response_model=HealthResponse tells FastAPI to validate and
# serialize the return value against the HealthResponse schema.
#
# Why /health and not /api/health?
# Nginx will handle the /api prefix stripping before the request
# reaches this container. From FastAPI's perspective the path
# is always /health. This keeps the FastAPI app unaware of how
# it is deployed — it does not need to know it sits behind a proxy.
# This is the correct separation of concerns.
@app.get("/health", response_model=HealthResponse)
def health_check():
    # Returns the status and which instance answered.
    # When we run repeated curl calls through Nginx, we will see
    # "instance" alternate between "fastapi_a" and "fastapi_b"
    # in the response — confirming round-robin load balancing works.
    return HealthResponse(status="ok", instance=INSTANCE_ID)


# ── Root endpoint ─────────────────────────────────────────────────
# A minimal root handler so hitting /api/ in the browser returns
# something readable instead of a 404.
# Not part of the Definition of Done — just reduces confusion.
@app.get("/")
def root():
    return {
        "message": "IDP Platform API",
        "instance": INSTANCE_ID,
        "docs": "/docs"
    }