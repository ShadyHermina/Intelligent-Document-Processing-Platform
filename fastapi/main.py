# fastapi/main.py
#
# Application entry point.
#
# Three responsibilities only:
#   1. Define the lifespan context manager (startup + shutdown)
#   2. Create the FastAPI app object with lifespan attached
#   3. Register routers
#
# No business logic. No SQL. No endpoint handlers.
# All of that lives in routers/ and dependencies/.

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from core.config import get_settings
from core.database import init_db_pool, close_db_pool, get_pool

from routers.session import router as session_router
from routers.documents import router as documents_router

from sentence_transformers import SentenceTransformer
from core.reranker import load_reranker
from routers.chat import router as chat_router

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.

    Everything before yield runs once when uvicorn starts the application,
    before the first request is accepted.

    Everything after yield runs once when uvicorn receives a shutdown signal
    (SIGTERM from docker compose down, or Ctrl+C in development), after the
    last in-flight request completes.

    Why asynccontextmanager and not @app.on_event("startup")?

    @app.on_event("startup") and @app.on_event("shutdown") are the older
    FastAPI pattern. They still work but are deprecated as of FastAPI 0.93.
    The lifespan context manager replaces both with a single function that
    keeps startup and shutdown logic visually together. It also makes the
    relationship between setup and teardown explicit — the same function
    that opens a resource closes it, in the same visual scope.

    Why does lifespan receive app as a parameter?

    FastAPI passes the app instance to the lifespan function automatically.
    We forward it to init_db_pool() and close_db_pool() so they can attach
    and retrieve the pool from app.state without importing the app object
    directly (which would create a circular import).
    """
    settings = get_settings()

    print(
        f"[startup] IDPP API starting — "
        f"instance={settings.instance_id} "
        f"env={settings.app_env}"
    )

    # ---- Startup ----

    await init_db_pool(app)
    # init_db_pool opens 2-5 asyncpg connections to PostgreSQL and stores
    # the pool on app.state.db_pool. From this point forward, any request
    # handler that calls get_pool(app) will receive the pool immediately.
    # If this call fails (wrong credentials, PostgreSQL unreachable), the
    # exception propagates here and uvicorn aborts startup — the application
    # never opens for traffic, which is the correct behavior. A server that
    # starts without a database connection would serve 500 errors to every
    # request until someone noticed.


    app.state.st_model = SentenceTransformer("all-MiniLM-L6-v2")
    # Load the local sentence embedding model used by Level 2 of the
    # three-level chunking pipeline (semantic topic-transition detection).
    #
    # Why import inside lifespan and not at module level?
    # sentence_transformers pulls in PyTorch — a large library. Importing
    # it at module level would slow every script that imports from main.py.
    # Importing inside lifespan defers the cost to startup only.
    #
    # Why app.state and not a module-level variable?
    # app.state is scoped to the app instance — test cases that create
    # isolated app instances get isolated model state. A module-level
    # variable would be shared across all instances in the same process.
    #
    # This runs before yield — before FastAPI accepts any requests.
    # Every IngestorAgent call finds app.state.st_model already populated.
    print("[startup] SentenceTransformer loaded — all-MiniLM-L6-v2")

    load_reranker(settings.reranker_model)
    # Load the cross-encoder reranker model into core.reranker._reranker.
    # Called here — once, at startup, before any request is accepted.
    # settings.reranker_model resolves to the RERANKER_MODEL env var,
    # defaulting to "cross-encoder/ms-marco-MiniLM-L-6-v2".
    # On first startup this downloads ~90MB from HuggingFace Hub.
    # Subsequent restarts load from the container's HuggingFace cache.
    # load_reranker() emits two log lines visible in docker compose logs:
    #   [startup] Loading reranker model: cross-encoder/...
    #   [startup] Reranker model loaded successfully
    print(f"[startup] CrossEncoder loaded — {settings.reranker_model}")

    print("[startup] complete — accepting requests")

    yield
    # ---- Application runs here ----
    # uvicorn is now accepting requests.
    # This yield suspends the lifespan coroutine until shutdown is triggered.
    # The event loop is free to handle requests while we wait here.

    # ---- Shutdown ----
    print("[shutdown] IDPP API shutting down ...")
    await close_db_pool(app)
    # close_db_pool waits for any in-flight queries to complete, then sends
    # proper disconnect messages to PostgreSQL before closing the TCP
    # connections. This prevents PostgreSQL from treating our connections
    # as crashed clients and holding them open unnecessarily.

    print("[shutdown] complete")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

settings = get_settings()
# We call get_settings() here to read the instance_id for the app title.
# Because of lru_cache, this is the same Settings object that lifespan
# and all routers will use — no re-reading of environment variables.

app = FastAPI(
    title="Intelligent Document Processing Platform — API",
    version="0.3.0",
    description=(
        "Secure multi-tenant document processing API. "
        f"Instance: {settings.instance_id}"
    ),
    lifespan=lifespan,
    # lifespan= is how we attach our startup/shutdown logic to this app.
    # FastAPI stores the lifespan and calls it when uvicorn starts and stops.
    # Without this parameter, the lifespan function defined above would
    # exist but never be called — a silent misconfiguration.
)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(session_router,   prefix="/session",   tags=["session"])
app.include_router(documents_router, prefix="/documents", tags=["documents"])
app.include_router(chat_router, prefix="/ws", tags=["chat"])



# ---------------------------------------------------------------------------
# Built-in endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """
    Liveness probe used by Docker Compose health checks and Nginx.

    Returns instance identity so we can confirm round-robin load
    balancing is distributing requests across both FastAPI instances.

    Why async def instead of def?

    FastAPI can handle both sync and async endpoint functions. We use
    async def here for consistency with the rest of the codebase — all
    endpoints that do I/O will be async, so making the health check
    async establishes the pattern even though this specific endpoint
    does no I/O.

    Why not check database connectivity here?

    A health check that queries the database couples liveness to database
    availability. If the database goes down briefly, all health checks
    fail, Nginx marks both FastAPI instances as unhealthy, and the entire
    platform goes dark — even though FastAPI itself is running fine.
    Liveness (is the process alive?) and readiness (can it serve traffic?)
    are separate concerns. This endpoint answers liveness only.
    In Phase 9 (Observability) we add a separate /ready endpoint that
    checks database connectivity.
    """
    pool = get_pool(app)
    # We call get_pool() here not to use a connection, but to confirm
    # the pool was successfully initialized at startup. If get_pool()
    # raises RuntimeError, the health check returns 500, which correctly
    # signals that startup did not complete successfully.
    # We do not acquire a connection — that would consume a pool slot
    # on every health check poll (every 10 seconds per docker-compose.yml).

    return {
        "status": "ok",
        "instance": settings.instance_id,
        "pool_size": pool.get_size(),
        # pool.get_size() returns the current number of open connections.
        # Seeing this in the health response confirms the pool is alive
        # and tells us how many connections are currently established.
    }


@app.get("/")
async def root():
    return {
        "message": "IDP Platform API",
        "instance": settings.instance_id,
        "docs": "/docs",
    }

# ---------------------------------------------------------------------------
# Temporary debug endpoint
# ---------------------------------------------------------------------------

@app.get("/debug/headers")
async def debug_headers(request: Request):
    """
    Returns all headers FastAPI received from Nginx.
    Used to verify real client IP is being forwarded correctly.
    Remove this endpoint after Phase 4 verification is complete.
    """
    return {
        "headers": dict(request.headers),
        "client_host": request.client.host,
    }

from fastapi import WebSocket

@app.websocket("/echo")
async def websocket_echo(websocket: WebSocket):
    """
    Temporary WebSocket echo endpoint for Phase 4 verification only.
    Accepts a connection, receives one message, echoes it back, then closes.
    Remove this endpoint after Phase 4 verification is complete.
    """
    await websocket.accept()
    message = await websocket.receive_text()
    await websocket.send_text(f"echo: {message}")
    await websocket.close()
# ---------------------------------------------------------------------------