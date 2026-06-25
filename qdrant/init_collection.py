# qdrant/init_collection.py
#
# One-time setup script. Run manually inside a container on idp_network
# to create the Qdrant collection and its tenant_id payload index.
#
# This script is NOT imported by any other module. shared/qdrant_store.py
# is the reusable client for all other phases. Keeping setup logic here,
# separate from the shared client, prevents collection creation from ever
# being called on a runtime import path — which would silently wipe all
# stored embeddings.

import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

# Read from environment. Compose injects these from .env into the
# container at startup — no python-dotenv needed here. python-dotenv
# is for scripts running on the host outside of Compose, where the
# shell does not already have these vars injected.
QDRANT_HOST = os.environ["QDRANT_HOST"]
QDRANT_PORT = int(os.environ["QDRANT_PORT"])

# Mirrors the Postgres table name "chunks" — both store two halves of
# the same logical entity. A chunk row in Postgres holds text, metadata,
# and position. Its corresponding point in Qdrant holds the vector.
# They are linked by the same chunk UUID used as the point ID.
COLLECTION_NAME = "document_chunks"

# Fixed by text-embedding-3-small's output dimensionality. Not a tunable.
# A wrong value here causes an immediate, loud dimension-mismatch error
# on every insert in Phase 5 — fail-fast, not silent data corruption.
VECTOR_SIZE = 1536

# Cosine measures the angle between vectors, not their magnitude.
# Two chunks expressing the same meaning at different lengths score
# as similar because cosine is invariant to vector magnitude.
# Alternative: Dot product — marginally cheaper when vectors are
# pre-normalised to unit length (OpenAI's embeddings are), but silently
# gives wrong-feeling results if a future model is not pre-normalised.
# Cosine is robust to that swap. We pay a negligible cost for that safety.
DISTANCE_METRIC = Distance.COSINE


def main():
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT} ...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Check whether the collection already exists before attempting
    # to create it. This replaces the deprecated recreate_collection
    # method, which combined existence check + delete + create into
    # one opaque call. The explicit pattern is clearer about intent
    # and produces no deprecation warnings.
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists — deleting ...")
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted.")

    print(f"Creating collection '{COLLECTION_NAME}' ...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=DISTANCE_METRIC,
        ),
    )
    print(f"  size={VECTOR_SIZE}, distance={DISTANCE_METRIC}")

    # Payload index on tenant_id.
    # Without this, every tenant-filtered search is a full collection
    # scan — O(n) regardless of how many tenants exist. With the index,
    # Qdrant narrows to only that tenant's points before scoring vectors.
    # KEYWORD = exact-match index. Correct for ID fields.
    # TEXT = tokenised full-text index. Wrong for IDs — would allow
    # partial matches, which is both incorrect and a security risk here.
    print("Creating payload index on 'tenant_id' (keyword) ...")
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="tenant_id",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print("  Done.")

    # Verify by reading collection info back from Qdrant.
    # We do not trust that the calls above succeeded silently —
    # we confirm the actual state Qdrant reports.
    print("\n--- Verification ---")
    info = client.get_collection(COLLECTION_NAME)
    print(f"Vector size:     {info.config.params.vectors.size}")
    print(f"Distance metric: {info.config.params.vectors.distance}")
    print(f"Payload schema:  {info.payload_schema}")
    print("\nCollection ready.")


if __name__ == "__main__":
    main()