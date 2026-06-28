# test_embedding_store.py
# Verifies that embed_and_store() correctly:
#   - calls OpenAI to produce real embeddings
#   - writes all chunks to Qdrant with correct payload fields
#   - returns one point ID per chunk in the correct order
#
# Run from inside the fastapi_a container:
#   python /tmp/test_embedding_store.py

import asyncio
from shared.embedding_service import embed_and_store
from shared.qdrant_store import get_client

TENANT_ID   = "bd8c8de3-4a8e-48b9-9065-9ac08918a9c7"
DOCUMENT_ID = "doc-cairo-urban-001"

CHUNKS = [
    "Cairo is the capital of Egypt and one of Africa largest cities.",
    "The city population exceeded 10 million in the 2020 census.",
    "Urban sprawl has pushed development toward New Cairo and 6th of October City.",
    "The Nile River divides Cairo into east and west administrative zones.",
]


async def main():
    # ── Step 1: store chunks ──────────────────────────────────────
    print("Storing chunks...")
    point_ids = await embed_and_store(CHUNKS, TENANT_ID, DOCUMENT_ID)

    print("Stored " + str(len(point_ids)) + " points:")
    for i, pid in enumerate(point_ids):
        print("  chunk_index=" + str(i) + "  point_id=" + pid)

    # ── Step 2: retrieve first point and inspect payload ──────────
    print("")
    print("Retrieving first point from Qdrant to verify payload...")

    client = get_client()
    results = client.retrieve(
        collection_name="document_chunks",
        ids=[point_ids[0]],
        with_payload=True,
        with_vectors=False,
    )

    point = results[0]
    print("  id          : " + str(point.id))
    print("  tenant_id   : " + point.payload["tenant_id"])
    print("  document_id : " + point.payload["document_id"])
    print("  chunk_index : " + str(point.payload["chunk_index"]))
    print("  text        : " + point.payload["text"])

    # ── Step 3: assert all 4 points were written ──────────────────
    print("")
    assert len(point_ids) == 4, "Expected 4 point IDs, got " + str(len(point_ids))
    assert point.payload["tenant_id"]   == TENANT_ID,   "tenant_id mismatch"
    assert point.payload["document_id"] == DOCUMENT_ID, "document_id mismatch"
    assert point.payload["chunk_index"] == 0,           "chunk_index mismatch"
    assert point.payload["text"]        == CHUNKS[0],   "text mismatch"

    print("All assertions passed.")


asyncio.run(main())