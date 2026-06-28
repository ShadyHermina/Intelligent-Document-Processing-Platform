# test_tenant_isolation.py
# Verifies cross-tenant isolation with real OpenAI embeddings:
#   - querying with a different tenant_id returns zero results
#   - querying with the correct tenant_id returns results
#   - proves isolation is enforced at the Qdrant filter level
#
# Assumes test_embedding_store.py has already been run so the
# Cairo chunks are already stored under the seed tenant.
#
# Run from inside the fastapi_a container:
#   python /tmp/test_tenant_isolation.py

import asyncio
from shared.embedding_service import embed_and_query

# The tenant whose data is already in Qdrant from the store test
REAL_TENANT_ID = "bd8c8de3-4a8e-48b9-9065-9ac08918a9c7"

# A fabricated tenant_id that has never stored anything in Qdrant.
# We use a valid UUID format because the embedding service only checks
# that tenant_id is non-empty — format validation is not its job.
# Using a realistic value makes this a stronger test: we are not
# triggering any format-based rejection, we are proving that a
# legitimate-looking but wrong tenant_id returns nothing.
FOREIGN_TENANT_ID = "ffffffff-ffff-ffff-ffff-ffffffffffff"

QUERY = "What is the population of Cairo?"


async def main():
    print("=" * 60)
    print("TEST 1: Query with FOREIGN tenant_id")
    print("Expected: zero results")
    print("=" * 60)
    print("Query     : " + QUERY)
    print("Tenant    : " + FOREIGN_TENANT_ID)
    print("")

    foreign_results = await embed_and_query(
        query=QUERY,
        tenant_id=FOREIGN_TENANT_ID,
        top_k=4,
    )

    print("Results returned: " + str(len(foreign_results)))
    if len(foreign_results) == 0:
        print("Correct — foreign tenant sees no data.")
    else:
        print("ISOLATION FAILURE — foreign tenant saw these chunks:")
        for r in foreign_results:
            print("  " + r["payload"]["text"])

    print("")
    print("=" * 60)
    print("TEST 2: Query with REAL tenant_id")
    print("Expected: results returned")
    print("=" * 60)
    print("Query     : " + QUERY)
    print("Tenant    : " + REAL_TENANT_ID)
    print("")

    real_results = await embed_and_query(
        query=QUERY,
        tenant_id=REAL_TENANT_ID,
        top_k=4,
    )

    print("Results returned: " + str(len(real_results)))
    for rank, result in enumerate(real_results):
        print("  Rank " + str(rank + 1))
        print("    score       : " + str(round(result["score"], 4)))
        print("    chunk_index : " + str(result["payload"]["chunk_index"]))
        print("    text        : " + result["payload"]["text"])
        print("")

    # ── Assertions ────────────────────────────────────────────────
    assert len(foreign_results) == 0, (
        "ISOLATION FAILURE: foreign tenant got "
        + str(len(foreign_results))
        + " results"
    )

    assert len(real_results) > 0, (
        "REGRESSION: real tenant got zero results — "
        "were the chunks deleted?"
    )

    # Every result must belong to the real tenant
    for result in real_results:
        assert result["payload"]["tenant_id"] == REAL_TENANT_ID, (
            "tenant_id mismatch in result: "
            + result["payload"]["tenant_id"]
        )

    print("All assertions passed.")
    print("Cross-tenant isolation confirmed with real OpenAI embeddings.")


asyncio.run(main())