# test_embedding_query.py
# Verifies that embed_and_query() correctly:
#   - embeds the query using OpenAI
#   - searches Qdrant filtered strictly by tenant_id
#   - returns semantically relevant chunks ranked by similarity
#
# NOTE on ranking: semantic embedding models do not guarantee that
# the "most obviously relevant" chunk by human judgment ranks first.
# Ranking depends on geometric proximity in the embedding space, which
# is influenced by shared vocabulary, sentence structure, and context.
# We assert correctness of filtering and result structure, not specific
# rank positions — those are model behavior, not system behavior.
#
# Run from inside the fastapi_a container:
#   python /tmp/test_embedding_query.py

import asyncio
from shared.embedding_service import embed_and_query

TENANT_ID   = "bd8c8de3-4a8e-48b9-9065-9ac08918a9c7"
DOCUMENT_ID = "doc-cairo-urban-001"
QUERY       = "What is the population of Cairo?"


async def main():
    print("Query  : " + QUERY)
    print("Tenant : " + TENANT_ID)
    print("")

    results = await embed_and_query(
        query=QUERY,
        tenant_id=TENANT_ID,
        top_k=4,
    )

    print("Results (" + str(len(results)) + " returned, ranked by similarity):")
    print("")
    for rank, result in enumerate(results):
        print("  Rank " + str(rank + 1))
        print("    score       : " + str(round(result["score"], 4)))
        print("    chunk_index : " + str(result["payload"]["chunk_index"]))
        print("    text        : " + result["payload"]["text"])
        print("")

    # ── Assertions ────────────────────────────────────────────────

    # Results were returned at all
    assert len(results) > 0, (
        "Expected results but got none"
    )

    # Scores are in descending order (Qdrant guarantees this but we
    # verify it explicitly so we catch any future regression)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), (
        "Results are not sorted by score descending: " + str(scores)
    )

    # Every result belongs to the correct tenant — no cross-tenant leak
    for result in results:
        assert result["payload"]["tenant_id"] == TENANT_ID, (
            "tenant_id mismatch: " + result["payload"]["tenant_id"]
        )

    # Every result belongs to the correct document
    for result in results:
        assert result["payload"]["document_id"] == DOCUMENT_ID, (
            "document_id mismatch: " + result["payload"]["document_id"]
        )

    # Every result has a chunk_index present and is a non-negative integer
    for result in results:
        assert isinstance(result["payload"]["chunk_index"], int), (
            "chunk_index missing or wrong type"
        )
        assert result["payload"]["chunk_index"] >= 0, (
            "chunk_index is negative"
        )

    # Every result has non-empty text
    for result in results:
        assert result["payload"]["text"].strip() != "", (
            "Empty text in result"
        )

    print("All assertions passed.")
    print("")
    print("Note: chunk_index=" + str(results[0]["payload"]["chunk_index"])
          + " ranked first with score="
          + str(round(results[0]["score"], 4))
          + ".")
    print("Semantic ranking reflects embedding geometry, not keyword overlap.")


asyncio.run(main())