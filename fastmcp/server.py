# server.py — FastMCP server (Phase 8)
#
# Three tools the Phase 9 RAG chatbot LLM will call at runtime to answer
# user questions about their documents:
#
#   search_documents      — structured metadata query   (PostgreSQL)
#   query_knowledge_base   — semantic similarity search  (Qdrant)
#   get_document_summary   — one document's full detail  (PostgreSQL)
#
# These are the ONLY MCP tools in the system. Everything else in the
# pipeline is plain backend logic. MCP tools exist for LLM-driven,
# runtime tool selection — the LLM decides which one to call based on
# what the user asked.
#
# --------------------------------------------------------------------------
# tenant_id contract (READ THIS)
# --------------------------------------------------------------------------
# Every tool takes tenant_id as its FIRST parameter. tenant_id is a TRUSTED,
# SERVER-INJECTED value. It is supplied by the FastAPI session layer on every
# call — never by the user and never by the LLM.
#
# The LLM never sees this function's signature. In Phase 9, FastAPI builds
# its OWN tool schema for OpenAI that omits tenant_id entirely, so the model
# cannot supply it. FastAPI resolves tenant_id from the authenticated session
# and injects it before calling these tools over HTTP. tenant_id being a plain
# parameter here is therefore an internal service contract between two of our
# own containers (fastapi -> fastmcp), not an LLM-facing input.
#
# Every database and vector query below filters on tenant_id. That filter is
# the mechanism that makes these tools tenant-safe.

import os
import json
from typing import Optional

import asyncpg
from fastmcp import FastMCP

from shared.embedding_service import embed_and_query

# FastMCP("name") creates the MCP server instance. The name is the server's
# identifier in the MCP protocol handshake. Unchanged from the stub.
mcp = FastMCP("IDP-MCP-Server")

# INSTANCE_ID: kept from the stub for consistency with the FastAPI pattern
# and for future logging of which container handled a call.
INSTANCE_ID = os.getenv("INSTANCE_ID", "fastmcp")


# ==========================================================================
# PostgreSQL connection helper
# ==========================================================================
# We open a FRESH asyncpg connection per tool call and close it in a finally
# block. No connection pool.
#
# Why no pool?
#   The MCP server is stateless by design (see architecture notes: it holds
#   no session state). A per-call connection matches that statelessness and
#   avoids coupling a pool to FastMCP's internally-managed event loop, which
#   we have not verified exposes a safe startup hook in this pinned version.
#   The cost is one connection open/close per call — a few milliseconds,
#   irrelevant at Phase 8 call volumes. If Phase 9 profiling shows this
#   matters, promoting to a pool is a localized change to this one helper.
#
# Why read os.environ directly?
#   The MCP server has no FastAPI config module and does not need one. The
#   fastmcp container already receives all five POSTGRES_* variables from
#   docker-compose. Reading them here directly is the same pattern the
#   project's migration scripts already use.

async def _connect() -> asyncpg.Connection:
    """
    Open and return a single asyncpg connection to PostgreSQL.

    Connection parameters are read from the environment variables injected
    into the fastmcp container by docker-compose. os.environ[...] (not
    os.getenv) is used deliberately: a missing variable raises KeyError
    immediately, which is the correct fail-loud behaviour for a missing
    connection parameter — better than a confusing timeout later.

    The caller is responsible for closing the returned connection. Every
    caller below does so inside a try/finally so the connection is released
    even if the query raises.
    """
    user     = os.environ["POSTGRES_USER"]
    password = os.environ["POSTGRES_PASSWORD"]
    database = os.environ["POSTGRES_DB"]
    host     = os.environ["POSTGRES_HOST"]
    port     = int(os.environ["POSTGRES_PORT"])
    # int(...) because asyncpg wants an int port. The env var arrives as the
    # string "5432"; PostgreSQL rejects a string port, so we coerce here.

    dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    # We never log the DSN — it contains the plaintext password.

    return await asyncpg.connect(dsn=dsn)
    # await because opening a TCP connection and authenticating is I/O.
    # asyncpg.connect returns a single Connection (not a pool).


# ==========================================================================
# Tool 1 — search_documents
# ==========================================================================

@mcp.tool()
async def search_documents(
    tenant_id: str,
    doc_type: Optional[str] = None,
    status: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> list[dict]:
    """
    Search a tenant's documents by structured metadata.

    Use this when the user asks WHICH documents exist or asks about document
    properties — for example "show me all contracts", "what documents were
    uploaded this month", "which invoices are still processing". This is a
    structured metadata query, not a content search.

    Parameters
    ----------
    tenant_id : str
        TRUSTED, server-injected tenant UUID. Not supplied by the LLM.
        Every returned row belongs to this tenant only.
    doc_type : str, optional
        Filter by classification: contract | invoice | claim | report | other.
    status : str, optional
        Filter by pipeline status: pending | classified | embedded.
    date_from : str, optional
        Lower bound on upload date (inclusive), ISO format e.g. "2025-01-01".
    date_to : str, optional
        Upper bound on upload date (inclusive), ISO format e.g. "2025-12-31".

    Returns
    -------
    list[dict]
        One dict per matching document with keys: id, filename, file_size,
        mime_type, status, doc_type, uploaded_at, processed_at. Timestamps
        are ISO-8601 strings. Empty list if nothing matches.
    """
    # We build the WHERE clause dynamically because the four filters are all
    # optional. tenant_id is ALWAYS present, so it is condition number one.
    #
    # conditions holds SQL fragments; params holds the matching values.
    # $1, $2, ... are asyncpg's positional placeholders. We NEVER interpolate
    # values into the SQL string with f-strings — that would be a SQL
    # injection hole. asyncpg substitutes params safely at execution time.
    conditions = ["tenant_id = $1"]
    params: list = [tenant_id]

    # Each optional filter, when provided, appends one condition and one
    # param. len(params) + 1 computes the next placeholder number so the
    # $N always lines up with the param's position in the list.
    if doc_type is not None:
        conditions.append(f"doc_type = ${len(params) + 1}")
        params.append(doc_type)

    if status is not None:
        conditions.append(f"status = ${len(params) + 1}")
        params.append(status)

    if date_from is not None:
        conditions.append(f"uploaded_at >= ${len(params) + 1}::timestamptz")
        params.append(date_from)

    if date_to is not None:
        conditions.append(f"uploaded_at <= ${len(params) + 1}::timestamptz")
        params.append(date_to)
    # NOTE on the f-strings above: they interpolate only the PLACEHOLDER
    # NUMBER (a controlled integer we compute), never any user value. The
    # actual values live in params and are bound safely by asyncpg.
    # ::timestamptz casts the ISO date string to a timestamp so the
    # comparison against the timestamptz column is well-typed.

    where_clause = " AND ".join(conditions)
    # Joins the fragments into e.g. "tenant_id = $1 AND doc_type = $2".

    query = f"""
        SELECT
            id,
            filename,
            file_size,
            mime_type,
            status,
            doc_type,
            uploaded_at,
            processed_at
        FROM documents
        WHERE {where_clause}
        ORDER BY uploaded_at DESC
    """
    # Column names are the REAL documents schema (base table + Phase 6
    # additions), verified against schema.sql and phase6_migrate.py:
    # filename (not "original_filename"), file_size, mime_type exist;
    # doc_type was added in Phase 6. ORDER BY newest-first is the natural
    # default for "show me my documents".

    conn = await _connect()
    try:
        rows = await conn.fetch(query, *params)
        # conn.fetch runs the SELECT and returns a list of asyncpg Record
        # objects. *params unpacks our value list into the positional
        # arguments that fill $1, $2, ...
    finally:
        await conn.close()
        # Always close, whether the query succeeded or raised. This releases
        # the connection cleanly instead of leaving PostgreSQL to time it out.

    # Convert each asyncpg Record into a plain dict the LLM layer can consume.
    # We stringify id and the timestamps: UUID and datetime objects are not
    # JSON-serialisable as-is, and every layer in this app works with string
    # UUIDs and ISO timestamps.
    results = []
    for row in rows:
        results.append({
            "id":           str(row["id"]),
            "filename":     row["filename"],
            "file_size":    row["file_size"],
            "mime_type":    row["mime_type"],
            "status":       row["status"],
            "doc_type":     row["doc_type"],
            "uploaded_at":  row["uploaded_at"].isoformat()  if row["uploaded_at"]  else None,
            "processed_at": row["processed_at"].isoformat() if row["processed_at"] else None,
        })
    return results


# ==========================================================================
# Tool 2 — query_knowledge_base
# ==========================================================================

@mcp.tool()
async def query_knowledge_base(
    tenant_id: str,
    query: str,
    top_k: int = 20,
) -> list[dict]:
    """
    Search a tenant's document CONTENT by semantic similarity.

    Use this when the user asks what the documents SAY — for example "what do
    our contracts say about termination", "find clauses about payment terms",
    "summarize the liability sections". This embeds the query and finds the
    most semantically similar chunks in the vector store.

    Parameters
    ----------
    tenant_id : str
        TRUSTED, server-injected tenant UUID. Not supplied by the LLM.
        Only this tenant's chunks are searched.
    query : str
        The natural-language question or phrase to search for.
    top_k : int, default 20
        Number of candidate chunks to return. Defaults to 20 — higher than
        the 3-5 the LLM ultimately needs, because Phase 9 adds a reranker
        that selects the best few from this larger candidate pool. A larger
        pool reranked down is more accurate than a small pool used directly.

    Returns
    -------
    list[dict]
        Ranked chunks, most similar first. Each dict has: id (Qdrant point
        id), score (cosine similarity 0-1), and payload (the chunk's stored
        metadata: text, document_id, chunk_index, location_index,
        section_label, image_present, doc_type, file_type, tenant_id).
        Empty list if the tenant has no matching content.
    """
    return await embed_and_query(
        query     = query,
        tenant_id = tenant_id,
        top_k     = top_k,
    )
    # We reuse the existing shared function verbatim. embed_and_query embeds
    # the query text via OpenAI (text-embedding-3-small) and calls
    # search_with_tenant, which filters strictly by tenant_id at the Qdrant
    # layer. We do not re-implement embedding or search here — the shared
    # module is the single source of truth for that logic, and it is async
    # because it awaits AsyncOpenAI, which is why this tool is async too.


# ==========================================================================
# Tool 3 — get_document_summary
# ==========================================================================

@mcp.tool()
async def get_document_summary(
    tenant_id: str,
    document_id: str,
) -> Optional[dict]:
    """
    Retrieve one specific document's full metadata and extracted entities.

    Use this when the user asks about a SPECIFIC document — for example "tell
    me about the Acme contract", "what entities were extracted from invoice
    1234", "what is the status of the document I just uploaded".

    Parameters
    ----------
    tenant_id : str
        TRUSTED, server-injected tenant UUID. Not supplied by the LLM.
    document_id : str
        UUID of the document to summarise.

    Returns
    -------
    dict or None
        The document's metadata (id, filename, file_size, mime_type, status,
        doc_type, uploaded_at, processed_at) plus extracted_entities (the
        parsed JSON object written by Agent 2). Returns None if no document
        with that id belongs to this tenant — this is how cross-tenant
        access is denied: a foreign document_id simply matches zero rows.
    """
    query = """
        SELECT
            id,
            filename,
            file_size,
            mime_type,
            status,
            doc_type,
            extracted_entities,
            uploaded_at,
            processed_at
        FROM documents
        WHERE id = $1 AND tenant_id = $2
    """
    # The two-condition WHERE is the cross-tenant guard. Even if the LLM (or
    # a bug upstream) supplies a document_id belonging to another tenant, the
    # AND tenant_id = $2 makes it match nothing. The tool returns None — not
    # an error, not another tenant's data. Empty by construction.

    conn = await _connect()
    try:
        row = await conn.fetchrow(query, document_id, tenant_id)
        # fetchrow returns a single Record, or None if no row matches.
    finally:
        await conn.close()

    if row is None:
        return None
        # No such document for this tenant. The caller distinguishes "not
        # found / not yours" from a real result by this None.

    # extracted_entities is a JSONB column. Agent 2 wrote it via json.dumps(),
    # and asyncpg returns JSONB as a raw JSON STRING (no automatic decoding is
    # configured on this connection). So we parse it back into a dict here.
    # If it is NULL (document not yet classified), default to an empty dict.
    raw_entities = row["extracted_entities"]
    if raw_entities is None:
        entities = {}
    else:
        entities = json.loads(raw_entities)

    return {
        "id":                 str(row["id"]),
        "filename":           row["filename"],
        "file_size":          row["file_size"],
        "mime_type":          row["mime_type"],
        "status":             row["status"],
        "doc_type":           row["doc_type"],
        "extracted_entities": entities,
        "uploaded_at":        row["uploaded_at"].isoformat()  if row["uploaded_at"]  else None,
        "processed_at":       row["processed_at"].isoformat() if row["processed_at"] else None,
    }


# ==========================================================================
# Server startup
# ==========================================================================
# Preserved EXACTLY from the stub — same transport, host, and port.
#
# transport="streamable-http": the network-capable transport that lets other
#   containers (FastAPI) call these tools over HTTP. stdio would require
#   running as a subprocess of the caller, incompatible with our separate
#   containers.
# host="0.0.0.0": bind all interfaces so Docker networking can route to us.
# port=8080: matches EXPOSE in the Dockerfile and the healthcheck in compose.
if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8080,
    )