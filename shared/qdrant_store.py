# shared/qdrant_store.py
#
# Reusable Qdrant client module. Imported by all phases that need to
# read from or write to the vector store:
#   Phase 5  — embedding service writes vectors via upsert_point()
#   Phase 8  — FastMCP tools read vectors via search_with_tenant()
#   Phase 9  — RAG pipeline reads vectors via search_with_tenant()
#
# Import path from any container with /app on PYTHONPATH:
#   from shared.qdrant_store import get_client, upsert_point, search_with_tenant
#
# This module never creates or deletes collections. That responsibility
# belongs exclusively to qdrant/init_collection.py. Keeping setup logic
# out of this module prevents accidental collection recreation on import.

import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

# Read from environment — injected by Compose from .env at container
# startup. os.environ raises KeyError immediately if either var is
# missing, which is the correct behaviour: a missing connection var
# is a configuration error that should fail loud and early, not
# produce a confusing connection timeout later.
QDRANT_HOST = os.environ["QDRANT_HOST"]
QDRANT_PORT = int(os.environ["QDRANT_PORT"])

COLLECTION_NAME = "document_chunks"

# Module-level singleton. The client is created once when this module
# is first imported and reused for every subsequent call. Creating a
# new client per request would open a new HTTP connection each time —
# wasteful and slower. A singleton reuses the underlying connection pool.
# This is safe because QdrantClient is thread-safe.
_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """
    Return the shared QdrantClient singleton, initialising it on first call.

    Why a function rather than a bare module-level assignment?
    A bare assignment runs at import time with no error handling. A
    function defers initialisation until first use and makes the
    connection point explicit and testable. Any phase that needs the
    raw client for an operation not covered by this module can call
    get_client() directly.
    """
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def upsert_point(
    point_id: str,
    vector: list[float],
    tenant_id: str,
    payload: dict,
) -> None:
    """
    Insert or update a single vector point in the collection.

    Args:
        point_id:  The chunk's UUID, shared with the Postgres chunks table.
                   Using the same UUID in both stores means a single ID
                   unambiguously identifies a chunk in either system —
                   no separate mapping table needed.
        vector:    The embedding vector. Must be exactly 1536 floats.
                   Qdrant rejects any vector with a different dimension
                   immediately — fail-fast, not silent corruption.
        tenant_id: The tenant this chunk belongs to. Stored in the payload
                   and used by search_with_tenant() to enforce isolation.
                   Also indexed (KEYWORD) for performant filtering.
        payload:   Any additional metadata to store alongside the vector
                   (e.g. chunk_index, document_id, source filename).
                   tenant_id is merged into this dict before upsert so
                   it is always present regardless of what the caller passes.

    Why upsert and not insert?
    Upsert (update-or-insert) is idempotent — running the embedding
    pipeline twice on the same chunk overwrites the existing point rather
    than creating a duplicate or raising an error. This makes the pipeline
    safely re-runnable without manual deduplication.
    """
    client = get_client()

    # Merge tenant_id into payload unconditionally. The caller should
    # not be responsible for remembering to include it — tenant isolation
    # is a system-level guarantee, not a caller convention.
    full_payload = {**payload, "tenant_id": tenant_id}

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload=full_payload,
            )
        ],
    )


def search_with_tenant(
    query_vector: list[float],
    tenant_id: str,
    limit: int = 5,
) -> list[dict]:
    """
    Search for the most similar vectors, filtered strictly to one tenant.

    The tenant_id filter is not optional and cannot be bypassed by the
    caller. Every search in this system is tenant-scoped — there is no
    legitimate use case for a cross-tenant search in the application layer.
    If a future phase needs a cross-tenant admin search, it should call
    get_client() directly and construct its own query explicitly, making
    the bypass visible and intentional rather than accidental.

    Args:
        query_vector: The embedding of the user's query. Must be 1536 floats.
        tenant_id:    Only points with this tenant_id in their payload are
                      considered. Points belonging to other tenants are
                      invisible to this search — Qdrant's index enforces
                      this at the database layer, not in application code.
        limit:        Maximum number of results to return. Defaults to 5,
                      which is a standard RAG context window size — enough
                      chunks to give the LLM useful context without
                      overwhelming the prompt.

    Returns:
        A list of dicts, each containing the point's id, score, and payload.
        Score is a cosine similarity value between 0 and 1 — higher means
        more similar. The list is sorted by score descending (most similar
        first) by Qdrant before being returned.

    Why return dicts instead of raw ScoredPoint objects?
    Returning raw Qdrant SDK objects would couple every caller to the
    qdrant-client package's internal data structures. If the SDK changes
    its response shape in a future version, every caller breaks. Returning
    plain dicts means only this function needs to change — callers are
    insulated from SDK internals.
    """
    client = get_client()

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(value=tenant_id),
                )
            ]
        ),
        limit=limit,
    )

    return [
        {
            "id": hit.id,
            "score": hit.score,
            "payload": hit.payload,
        }
        for hit in results
    ]