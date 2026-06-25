# testing/test_isolation.py
#
# Phase 2 isolation test. Run once inside fastapi_a to prove that
# Qdrant's tenant_id filter prevents cross-tenant data access.
# Delete or ignore after Phase 2 is confirmed complete.

import uuid
from shared.qdrant_store import upsert_point, search_with_tenant

TENANT_A = "tenant-a-000-000-000"
TENANT_B = "tenant-b-000-000-000"

# Synthetic vector — real embeddings come in Phase 5.
# Values don't matter here; isolation is what we're testing.
test_vector = [1.0] + [0.0] * 1535
point_id = str(uuid.uuid4())

print(f"Inserting test point {point_id} for tenant A ...")
upsert_point(
    point_id=point_id,
    vector=test_vector,
    tenant_id=TENANT_A,
    payload={"source": "phase2_isolation_test", "chunk_index": 0},
)
print("  Inserted.")

print()
print("Searching as tenant A (should find 1 result) ...")
results_a = search_with_tenant(
    query_vector=test_vector,
    tenant_id=TENANT_A,
    limit=5,
)
print(f"  Results: {len(results_a)}")
for r in results_a:
    print(f"  id={r['id']} score={r['score']:.4f} payload={r['payload']}")

print()
print("Searching as tenant B (should find 0 results) ...")
results_b = search_with_tenant(
    query_vector=test_vector,
    tenant_id=TENANT_B,
    limit=5,
)
print(f"  Results: {len(results_b)}")

print()
if len(results_a) == 1 and len(results_b) == 0:
    print("ISOLATION CONFIRMED: tenant A sees its point, tenant B sees nothing.")
else:
    print("ISOLATION FAILED — investigate before proceeding.")