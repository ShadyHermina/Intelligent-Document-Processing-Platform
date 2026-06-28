# scripts/phase6_migrate.py
#
# Phase 6 schema migration — adds columns required by the document
# ingestion pipeline to the documents and chunks tables.
#
# Safety properties:
#   · Uses ALTER TABLE ... ADD COLUMN IF NOT EXISTS throughout.
#     Running this script twice produces the same result as running
#     it once. No error is raised if a column already exists.
#   · No tables are dropped. No existing rows are modified.
#     All new columns are nullable so existing rows remain valid.
#   · The seed tenant, existing documents, and Phase 5 test data
#     are completely unaffected.
#
# Run from the project root on the Windows host:
#   docker exec -it postgres psql -U idp_user -d idp_db -f /dev/stdin < scripts/phase6_migrate.py
#
# Or run as a Python script (preferred — gives clear per-statement output):
#   docker exec -it fastapi_a python /app/scripts/phase6_migrate.py
#
# Prerequisites:
#   · All containers running and healthy
#   · POSTGRES_* environment variables present (injected by Compose)

import asyncio
import os
import asyncpg


# ---------------------------------------------------------------------------
# Migration statements
# ---------------------------------------------------------------------------
#
# Each statement is a tuple of (description, SQL).
# The description is printed before execution so you can see exactly
# which statement is running if one fails.
#
# Order matters only when a later statement depends on an earlier one.
# Here every statement is independent, so order is documentation only.

MIGRATIONS = [
    # ── documents table ──────────────────────────────────────────────────

    (
        "documents: add file_hash column",
        """
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS file_hash TEXT;
        """,
        # SHA-256 hex digest of the raw uploaded file bytes.
        # Used by the upload endpoint to detect duplicate files before
        # any processing begins. Nullable because existing rows predate
        # this column and have no computed hash.
    ),
    (
        "documents: add doc_type column",
        """
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS doc_type TEXT;
        """,
        # Document classification written by Agent 2 after GPT-4o runs.
        # Values: contract | invoice | claim | report | other
        # Nullable because it does not exist until Agent 2 completes.
        # A CHECK constraint on allowed values is a Phase 9 hardening task.
    ),
    (
        "documents: add extracted_entities column",
        """
        ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS extracted_entities JSONB;
        """,
        # Structured entity extraction written by Agent 2.
        # Stores parties, dates, amounts, flags as a JSON object.
        # JSONB (binary JSON) rather than plain JSON because:
        #   · JSONB is stored in a decomposed binary format — faster to query
        #   · JSONB supports GIN indexing for containment operators (@>)
        #   · JSONB deduplicates keys and discards insignificant whitespace
        # Nullable until Agent 2 runs.
    ),

    # ── chunks table ─────────────────────────────────────────────────────

    (
        "chunks: add location_index column",
        """
        ALTER TABLE chunks
        ADD COLUMN IF NOT EXISTS location_index INTEGER;
        """,
        # Page number for PDF and DOCX. Sheet index (1-based) for XLSX.
        # Renamed from the naive "page_number" to be format-agnostic.
        # Nullable for rows created before Phase 6.
    ),
    (
        "chunks: add section_label column",
        """
        ALTER TABLE chunks
        ADD COLUMN IF NOT EXISTS section_label TEXT;
        """,
        # Heading text (PDF/DOCX) or "SheetName / row group N" (XLSX).
        # Falls back to "Body" when no heading is detected.
        # Enables section-scoped retrieval queries in Qdrant and PostgreSQL.
        # Nullable for rows created before Phase 6.
    ),
    (
        "chunks: add image_present column",
        """
        ALTER TABLE chunks
        ADD COLUMN IF NOT EXISTS image_present BOOLEAN;
        """,
        # TRUE when an image was detected adjacent to this chunk's text.
        # Used to flag chunks that describe a figure or diagram.
        # No multimodal embedding in Phase 6 — detection only.
        # Nullable for existing rows; application logic defaults to FALSE.
    ),
    (
        "chunks: add file_type column",
        """
        ALTER TABLE chunks
        ADD COLUMN IF NOT EXISTS file_type TEXT;
        """,
        # Source format: "pdf" | "docx" | "xlsx"
        # Carried into each chunk so retrieval queries can filter by
        # format without joining back to the documents table.
        # Nullable for rows created before Phase 6.
    ),
    (
        "chunks: add doc_type column",
        """
        ALTER TABLE chunks
        ADD COLUMN IF NOT EXISTS doc_type TEXT;
        """,
        # Document classification copied from the classifier result.
        # Denormalized onto each chunk so retrieval queries can filter
        # by document type (e.g. "find all contract chunks for tenant X")
        # without a join to the documents table.
        # Nullable until Agent 2 runs.
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_migrations() -> None:
    """
    Connect to PostgreSQL and execute every migration statement in order.

    Each statement is executed and confirmed individually.
    If any statement fails, execution stops immediately and the error
    is printed with the statement that caused it. Already-executed
    statements are not rolled back — IF NOT EXISTS means re-running
    the script from the top is always safe.
    """

    # Read connection parameters from environment variables.
    # These are injected by Docker Compose from .env into every container.
    # Running this script inside fastapi_a or fastapi_b guarantees they
    # are present. Running it outside Docker requires exporting them manually.
    user     = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    database = os.environ["POSTGRES_DB"]
    host     = os.environ["POSTGRES_HOST"]
    port     = int(os.environ["POSTGRES_PORT"])

    dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    # We never print the DSN — it contains the plaintext password.

    print(f"[migrate] Connecting to {host}:{port}/{database} as {user}")

    conn = await asyncpg.connect(dsn=dsn)
    # asyncpg.connect() opens a single connection — not a pool.
    # We use a single connection here because migrations run once,
    # serially, and do not need concurrent access. A pool would add
    # overhead with no benefit.

    print(f"[migrate] Connected. Running {len(MIGRATIONS)} migration(s).\n")

    try:
        for description, sql, *_ in MIGRATIONS:
            # *_ discards the comment string in the tuple — it exists
            # only as inline documentation and is not passed to asyncpg.

            print(f"  → {description}")
            await conn.execute(sql)
            print(f"    OK\n")
            # conn.execute() runs a single DDL statement and waits for
            # PostgreSQL to confirm it completed. DDL statements in
            # PostgreSQL are transactional — if the statement fails,
            # PostgreSQL rolls it back automatically. No manual rollback
            # needed here.

    except Exception as exc:
        print(f"\n[migrate] FAILED: {exc}")
        print("[migrate] Fix the error above and re-run. Already-applied")
        print("[migrate] statements are safe to re-run (IF NOT EXISTS).")
        raise
        # Re-raise so the process exits with a non-zero code,
        # making failure visible to any wrapper script or CI system.

    finally:
        await conn.close()
        # Always close the connection — whether migrations succeeded or failed.
        # This sends a proper disconnect to PostgreSQL rather than leaving
        # a dangling connection that PostgreSQL holds open until timeout.

    print("[migrate] All migrations applied successfully.")
    print("\nVerify with:")
    print("  docker exec -it postgres psql -U idp_user -d idp_db \\")
    print(r'  -c "\d documents"')
    print("  docker exec -it postgres psql -U idp_user -d idp_db \\")
    print(r'  -c "\d chunks"')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_migrations())
    # asyncio.run() creates a new event loop, runs run_migrations() to
    # completion, then closes the loop. This is the correct pattern for
    # a standalone async script — not inside a running FastAPI server.