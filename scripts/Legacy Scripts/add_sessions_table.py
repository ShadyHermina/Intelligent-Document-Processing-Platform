"""
Migration: add_sessions_table.py

Adds the sessions table to an existing IDPP PostgreSQL database.
Safe to run multiple times -- uses CREATE TABLE IF NOT EXISTS.

How to run (from project root in PowerShell):
    docker cp scripts/add_sessions_table.py fastapi_a:/tmp/add_sessions_table.py
    docker exec fastapi_a python /tmp/add_sessions_table.py
"""

import asyncio
import sys

import asyncpg


# ---------------------------------------------------------------------------
# 1. Connection constants
# ---------------------------------------------------------------------------
# These are not read from environment variables because this script is
# run inside the fastapi_a container, where the compose environment block
# now supplies these values. We read them from os.environ so that the
# single source of truth remains docker-compose.yml and .env -- not this
# script.
#
# We do NOT use python-dotenv here because inside the container the
# environment variables are already injected by Docker. dotenv is only
# needed when running Python outside of Docker (e.g. on your Windows host)
# where the shell does not have those variables set. Inside the container,
# os.environ already has them.

import os

POSTGRES_USER     = os.environ["POSTGRES_USER"]      # idp_user
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]  # idp_secret
POSTGRES_DB       = os.environ["POSTGRES_DB"]        # idp_db
POSTGRES_HOST     = os.environ["POSTGRES_HOST"]      # postgres
POSTGRES_PORT     = os.environ["POSTGRES_PORT"]      # 5432


# ---------------------------------------------------------------------------
# 2. SQL statements
# ---------------------------------------------------------------------------
CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT        PRIMARY KEY,
    tenant_id   UUID        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ NOT NULL
);
"""
# PRIMARY KEY on token:
#   Enforces uniqueness and auto-creates a B-tree index.
#   Every /api/session/me lookup queries by token -- this index
#   makes that lookup O(log n) instead of a full table scan.
#
# REFERENCES tenants(tenant_id) ON DELETE CASCADE:
#   Foreign key -- token must belong to a real tenant.
#   ON DELETE CASCADE -- if a tenant is deleted, their sessions
#   are deleted automatically. No orphaned sessions ever.
#
# TIMESTAMPTZ not TIMESTAMP:
#   Always store timestamps with timezone in PostgreSQL.
#   TIMESTAMP without timezone stores local time with no context --
#   ambiguous when servers change timezone or daylight saving occurs.
#   TIMESTAMPTZ stores UTC internally and converts on display.


CREATE_EXPIRES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at
    ON sessions(expires_at);
"""
# Why a separate index on expires_at when token already has one?
#
# The token index handles lookups: WHERE token = $1
# This index handles range scans: WHERE expires_at > now()
#
# The /api/session/me query uses BOTH filters together:
#   WHERE token = $1 AND expires_at > now()
# PostgreSQL will use the token index (more selective) and check
# expiry on the matched row -- so this index primarily benefits
# the future cleanup job:
#   DELETE FROM sessions WHERE expires_at < now()
# That DELETE scans by time range, not by token -- without this
# index it would be a full table scan every time the job runs.


# ---------------------------------------------------------------------------
# 3. Migration coroutine
# ---------------------------------------------------------------------------
async def run_migration() -> None:
    """
    Opens one direct connection to PostgreSQL, runs both DDL statements,
    then closes. We use a direct connection (asyncpg.connect) rather than
    a pool (asyncpg.create_pool) because this is a one-shot admin script --
    pools are for long-lived servers that handle many concurrent requests.
    """
    dsn = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )

    print(f"Connecting to PostgreSQL at {POSTGRES_HOST}:{POSTGRES_PORT} ...")
    print(f"Database : {POSTGRES_DB}")
    print(f"User     : {POSTGRES_USER}")

    try:
        conn = await asyncpg.connect(dsn)
    except Exception as e:
        print(f"\nERROR: Could not connect to PostgreSQL.")
        print(f"Detail : {e}")
        sys.exit(1)

    try:
        print("\nCreating sessions table ...")
        await conn.execute(CREATE_SESSIONS_TABLE)
        # conn.execute() runs a DDL statement and returns a status string
        # like "CREATE TABLE" or "CREATE TABLE (already exists)".
        # IF NOT EXISTS means it is safe to run this on a database that
        # already has the table -- it becomes a no-op, not an error.

        print("Creating index on sessions.expires_at ...")
        await conn.execute(CREATE_EXPIRES_INDEX)

        print("\nMigration complete.")
        print("  Table created : sessions")
        print("  Index created : idx_sessions_expires_at")

    except Exception as e:
        print(f"\nERROR: Migration failed during SQL execution.")
        print(f"Detail : {e}")
        sys.exit(1)

    finally:
        # finally block runs whether the try succeeded or raised an
        # exception. This guarantees the connection is always closed
        # and returned cleanly to PostgreSQL -- no dangling connections.
        await conn.close()
        print("Connection closed.")


# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # asyncio.run() is the standard way to execute a top-level async
    # function from a synchronous entry point. It:
    #   1. Creates a new event loop
    #   2. Runs the coroutine to completion
    #   3. Closes the event loop
    # Without asyncio.run(), calling run_migration() would return a
    # coroutine object without executing it -- a common Python async
    # beginner mistake.
    asyncio.run(run_migration())