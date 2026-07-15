#!/usr/bin/env python3
"""
scripts/remove_tenant.py
-------------------------
Permanently removes a tenant and ALL of its associated data from BOTH
PostgreSQL and Qdrant.

This deliberately touches two separate stores with no cross-store
transaction -- the same dual-write consistency gap already present in
the ingestion path (see fastapi/agents/classifier.py, and the "Consistency
between stores" category from the architecture review). If this script
is interrupted partway through, re-run it with the same tenant_id --
every deletion step here is idempotent (deleting something already gone
is a no-op, not an error), so retrying is always safe.

What gets deleted:
    - The tenant row itself                              (tenants)
    - All its documents                                  (documents)   -- via CASCADE
    - All chunks belonging to those documents             (chunks)      -- via CASCADE
    - All sessions issued for this tenant                 (sessions)    -- via CASCADE
    - All Qdrant vector points with matching tenant_id in their payload

What is preserved (deliberately -- matches existing schema.sql design,
not something this script changes):
    - audit_log rows referencing this tenant. The FK is
      ON DELETE SET NULL, not CASCADE -- the row survives, only
      audit_log.tenant_id becomes NULL. This keeps the audit trail
      intact even after the tenant that generated it is gone.

Usage:
    python scripts/remove_tenant.py <tenant_id>
    python scripts/remove_tenant.py <tenant_id> --yes   # skip the confirmation prompt

Expects:
    - Docker Desktop running, postgres + qdrant containers healthy
    - Qdrant's HTTP port published to the host (true by default --
      see the qdrant service's `ports:` block in docker-compose.yml,
      commented "development only")
    - `pip install qdrant-client==1.13.3` on whatever host runs this
      script (pinned to match fastapi/requirements.txt). Same situation
      as scripts/seed.py needing uuid6 on the host: this script runs
      outside Docker entirely, so it can't reuse a container's already-
      installed packages.
"""

import subprocess
import sys
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ------------------------------------------------------------
# Configuration -- mirrors scripts/seed.py's convention of hardcoded
# constants matching known .env values, rather than parsing .env here.
# ------------------------------------------------------------

CONTAINER_NAME    = "postgres"
PG_USER           = "idp_user"
PG_DATABASE       = "idp_db"

QDRANT_HOST       = "localhost"  # the published host port, not the in-network "qdrant" hostname
QDRANT_PORT       = 6333
QDRANT_COLLECTION = "document_chunks"


# ------------------------------------------------------------
# Postgres helpers (same subprocess-via-docker-exec pattern as seed.py)
# ------------------------------------------------------------

def run_sql(sql: str, description: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [
            "docker", "exec",
            "-i",
            CONTAINER_NAME,
            "psql",
            "-U", PG_USER,
            "-d", PG_DATABASE,
            "-v", "ON_ERROR_STOP=1",
            "-q",
            "-t",
            "-A",
            "-F", "|",
        ],
        input=sql,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[ERROR] {description} failed.")
        print(result.stderr)
        sys.exit(1)
    return result


def fetch_tenant(tenant_id: str):
    sql = f"""
SELECT id, name, is_active, created_at
FROM tenants
WHERE id = '{tenant_id}';
"""
    result = run_sql(sql, "Fetch tenant")
    output = result.stdout.strip()
    if not output:
        return None
    parts = output.split("|")
    return {
        "id": parts[0].strip(),
        "name": parts[1].strip(),
        "is_active": parts[2].strip(),
        "created_at": parts[3].strip(),
    }


def count_rows(table: str, tenant_column: str, tenant_id: str) -> int:
    sql = f"SELECT COUNT(*) FROM {table} WHERE {tenant_column} = '{tenant_id}';"
    result = run_sql(sql, f"Count {table}")
    return int(result.stdout.strip())


def delete_tenant_postgres(tenant_id: str) -> None:
    # A single DELETE on tenants cascades through documents -> chunks,
    # and through sessions, via the ON DELETE CASCADE foreign keys
    # already defined in the schema. audit_log uses ON DELETE SET NULL
    # instead -- deliberately preserved, not touched here.
    sql = f"DELETE FROM tenants WHERE id = '{tenant_id}';"
    run_sql(sql, "Delete tenant (cascades to documents/chunks/sessions)")


# ------------------------------------------------------------
# Qdrant helpers
# ------------------------------------------------------------

def qdrant_tenant_filter(tenant_id: str) -> Filter:
    # Same Filter/FieldCondition/MatchValue pattern already used by
    # shared/qdrant_store.py's search_with_tenant() -- tenant_id is a
    # keyword-indexed payload field (see qdrant/init_collection.py).
    return Filter(
        must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
    )


def count_qdrant_points(client: QdrantClient, tenant_id: str) -> int:
    result = client.count(
        collection_name=QDRANT_COLLECTION,
        count_filter=qdrant_tenant_filter(tenant_id),
        exact=True,
    )
    return result.count


def delete_qdrant_points(client: QdrantClient, tenant_id: str) -> None:
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=qdrant_tenant_filter(tenant_id),
    )


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/remove_tenant.py <tenant_id> [--yes]")
        sys.exit(1)

    tenant_id_raw = sys.argv[1]
    skip_confirm = "--yes" in sys.argv[2:] or "-y" in sys.argv[2:]

    try:
        tenant_id = str(uuid.UUID(tenant_id_raw))
    except ValueError:
        print(f"[ERROR] '{tenant_id_raw}' is not a valid UUID.")
        sys.exit(1)

    print("=" * 60)
    print("  IDPP Tenant Removal")
    print("=" * 60)

    print("[...] Looking up tenant...")
    tenant = fetch_tenant(tenant_id)
    if tenant is None:
        print(f"[ERROR] No tenant found with id {tenant_id}.")
        sys.exit(1)

    doc_count     = count_rows("documents", "tenant_id", tenant_id)
    chunk_count   = count_rows("chunks", "tenant_id", tenant_id)
    session_count = count_rows("sessions", "tenant_id", tenant_id)
    audit_count   = count_rows("audit_log", "tenant_id", tenant_id)

    print("[...] Connecting to Qdrant...")
    try:
        qclient = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        point_count = count_qdrant_points(qclient, tenant_id)
    except Exception as e:
        print(f"[ERROR] Could not reach Qdrant at {QDRANT_HOST}:{QDRANT_PORT}: {e}")
        print("        Check that the qdrant container is healthy and that its port")
        print("        is published to the host (docker-compose.yml, qdrant.ports).")
        sys.exit(1)

    print()
    print("The following will be PERMANENTLY deleted:")
    print(f"  Tenant:        {tenant['name']}  ({tenant['id']})")
    print(f"  Active:        {tenant['is_active']}   Created: {tenant['created_at']}")
    print(f"  Documents:     {doc_count}")
    print(f"  Chunks:        {chunk_count}")
    print(f"  Sessions:      {session_count}")
    print(f"  Qdrant points: {point_count}")
    print()
    print("The following will be PRESERVED but anonymized (tenant_id set to NULL):")
    print(f"  Audit log entries: {audit_count}")
    print()

    if not skip_confirm:
        confirmation = input(
            f'Type the tenant name exactly ("{tenant["name"]}") to confirm deletion, '
            f"or anything else to abort: "
        )
        if confirmation != tenant["name"]:
            print("[ABORTED] Confirmation text did not match. Nothing was deleted.")
            sys.exit(1)

    print()
    print("[...] Deleting Qdrant points...")
    delete_qdrant_points(qclient, tenant_id)

    print("[...] Deleting tenant from PostgreSQL (cascades to documents/chunks/sessions)...")
    delete_tenant_postgres(tenant_id)

    print("[...] Verifying deletion...")
    remaining_docs     = count_rows("documents", "tenant_id", tenant_id)
    remaining_chunks   = count_rows("chunks", "tenant_id", tenant_id)
    remaining_sessions = count_rows("sessions", "tenant_id", tenant_id)
    remaining_tenant   = fetch_tenant(tenant_id)
    remaining_points   = count_qdrant_points(qclient, tenant_id)

    problems = []
    if remaining_tenant is not None:
        problems.append("tenant row still exists")
    if remaining_docs != 0:
        problems.append(f"{remaining_docs} document rows still exist")
    if remaining_chunks != 0:
        problems.append(f"{remaining_chunks} chunk rows still exist")
    if remaining_sessions != 0:
        problems.append(f"{remaining_sessions} session rows still exist")
    if remaining_points != 0:
        problems.append(f"{remaining_points} Qdrant points still exist")

    if problems:
        print("[ERROR] Deletion did not fully complete:")
        for p in problems:
            print(f"        - {p}")
        print("        Re-run this script with the same tenant_id -- every deletion")
        print("        step here is idempotent and safe to retry.")
        sys.exit(1)

    print("[OK]    Tenant and all associated data removed successfully.")
    print(f"        {audit_count} audit_log entries preserved with tenant_id = NULL.")
    print()
    print("=" * 60)
    print("  Removal complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()