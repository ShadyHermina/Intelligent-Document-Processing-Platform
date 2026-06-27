# fastapi/core/database.py
#
# Owns the asyncpg connection pool lifecycle.
#
# Three responsibilities:
#   1. init_db_pool(app)  — create the pool at application startup
#   2. close_db_pool(app) — close the pool at application shutdown
#   3. get_pool()         — return the pool to any code that needs a connection
#
# Nothing in this file runs at import time.
# The pool is created only when init_db_pool() is explicitly called
# during the FastAPI lifespan startup event in main.py.

import asyncpg
from fastapi import FastAPI

from core.config import get_settings


async def init_db_pool(app: FastAPI) -> None:
    """
    Create the asyncpg connection pool and attach it to app.state.

    Called once during FastAPI lifespan startup — before the application
    begins accepting requests. By the time the first request arrives,
    the pool already exists and connections are ready to be borrowed.

    Why app.state and not a module-level variable?

    A module-level variable like `_pool = None` works but creates a
    subtle problem: the variable belongs to the module, not to the
    application instance. In testing, when you create multiple FastAPI
    app instances in the same process (common in pytest), they would all
    share and potentially corrupt the same module-level variable.

    app.state is owned by the specific FastAPI instance. Each app instance
    has its own state. Tests that create isolated app instances get
    isolated pools. Production code that has one app instance gets one pool.

    Why pass app as a parameter instead of importing it from main?

    Importing app from main.py would create a circular import:
      database.py imports from main.py
      main.py imports from database.py
    Python cannot resolve circular imports. Passing app as a parameter
    breaks the circle — database.py never needs to know about main.py.

    Parameters
    ----------
    app : FastAPI
        The application instance whose state will hold the pool.
    """
    settings = get_settings()
    # get_settings() returns the cached Settings object — no re-reading
    # of environment variables. This call costs the same as a dictionary
    # lookup because of the lru_cache we set up in config.py.

    dsn = (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}"
        f"/{settings.postgres_db}"
    )
    # The DSN is constructed here from typed Settings attributes.
    # settings.postgres_port is already an int (coerced by pydantic-settings)
    # but f-string interpolation converts it back to a string automatically.
    # We never log the DSN — it contains the plaintext password.

    app.state.db_pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=2,
        max_size=5,
    )
    # create_pool() must be awaited — it opens the initial connections
    # to PostgreSQL synchronously from asyncpg's perspective, but the
    # await allows the event loop to do other work if needed during
    # the TCP handshake and authentication with each connection.
    #
    # min_size=2:
    #   Keep at least 2 connections open at all times, even at zero load.
    #   The first request after a quiet period gets a connection instantly
    #   rather than waiting for a new connection to be established.
    #   Without min_size, the pool could close all connections during
    #   idle periods and then face connection-establishment latency on
    #   the next request.
    #
    # max_size=5:
    #   Never open more than 5 simultaneous connections from this instance.
    #   We have 2 FastAPI instances, so the total connection count is
    #   2 × 5 = 10 connections out of PostgreSQL's default limit of 100.
    #   If all 5 connections are in use, incoming requests wait in a
    #   queue rather than failing — this is correct behavior under load.
    #   Increase max_size only if profiling shows pool exhaustion.

    print(
        f"[database] Pool created — "
        f"min={2} max={5} "
        f"host={settings.postgres_host} "
        f"db={settings.postgres_db}"
    )
    # We print host and db but NOT the password or the full DSN.
    # This log line appears in `docker logs fastapi_a` at startup,
    # confirming the pool initialized successfully.
    # In later phases this becomes a proper structured log call.


async def close_db_pool(app: FastAPI) -> None:
    """
    Close the asyncpg connection pool gracefully at application shutdown.

    Called once during FastAPI lifespan shutdown — after the application
    stops accepting new requests but before the process exits.

    Why close explicitly instead of letting the process exit?

    When a Python process exits, open TCP connections are closed by the
    operating system eventually, but not immediately. PostgreSQL tracks
    open connections and has a maximum connection limit. If uvicorn is
    restarted rapidly (common during development), connections from the
    previous process may still appear open to PostgreSQL, consuming slots
    from the connection limit. Explicit closure sends a proper disconnect
    message that PostgreSQL processes immediately.

    pool.close() waits for all in-flight queries to complete before
    closing the connections. It does not kill queries mid-execution.

    Parameters
    ----------
    app : FastAPI
        The application instance whose pool will be closed.
    """
    pool = getattr(app.state, "db_pool", None)
    # getattr with a default of None handles the edge case where
    # init_db_pool() failed before assigning app.state.db_pool.
    # Without this, close_db_pool() would raise AttributeError during
    # shutdown, masking the original startup error in the logs.

    if pool is not None:
        await pool.close()
        print("[database] Pool closed.")


def get_pool(app: FastAPI) -> asyncpg.Pool:
    """
    Return the connection pool attached to the given app instance.

    This function is called by request handlers and dependencies that
    need to run a database query. The caller receives the pool object
    and uses it as an async context manager to borrow one connection:

        pool = get_pool(app)
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT ...", value)
        # connection automatically returned to pool here

    Why return the pool instead of a connection directly?

    Returning a connection would require this function to be async
    (because acquiring a connection is async) and would force the caller
    to manage the connection lifecycle manually. Returning the pool keeps
    this function synchronous and delegates connection borrowing to the
    caller via the context manager pattern — which guarantees the
    connection is returned even if an exception is raised.

    Parameters
    ----------
    app : FastAPI
        The application instance whose pool will be returned.

    Returns
    -------
    asyncpg.Pool
        The connection pool. Guaranteed to exist if called after startup.

    Raises
    ------
    RuntimeError
        If called before init_db_pool() has run — i.e., before startup
        completes. This should never happen in normal operation but is
        a clear error for debugging misconfigured startup sequences.
    """
    pool = getattr(app.state, "db_pool", None)

    if pool is None:
        raise RuntimeError(
            "Database pool is not initialized. "
            "Ensure init_db_pool() was called during application startup."
        )

    return pool