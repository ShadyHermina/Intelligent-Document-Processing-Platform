#!/usr/bin/env python3
"""
scripts/seed.py
---------------
Inserts a test tenant into the running PostgreSQL container and
verifies the access_phrase_hash lookup works correctly.

Usage:
    python scripts/seed.py

Expects:
    - Docker Desktop running
    - Container named 'postgres' is healthy
    - Migration has already been applied (scripts/migrate.py)
"""

import subprocess
import sys
import hashlib

from uuid6 import uuid7
# Requires `pip install uuid6` on whatever host runs this script — it is
# NOT in any requirements.txt because this script runs outside Docker
# entirely (see module docstring: it shells out to `docker exec`, it
# doesn't run inside a container itself). If you set this project up on
# a fresh machine, `pip install uuid6` before running this script.

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

CONTAINER_NAME = "postgres"
PG_USER        = "idp_user"
PG_DATABASE    = "idp_db"

TEST_TENANT_NAME   = "Meridian Logistics"
TEST_PASSPHRASE    = "river-canyon-forge-42"
TEST_TENANT_ID     = str(uuid7())

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def hash_passphrase(passphrase: str) -> str:
    return hashlib.sha256(passphrase.encode("utf-8")).hexdigest()


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
            "-F", "|"
        ],
        input=sql,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"[ERROR] {description} failed.")
        print(result.stderr)
        sys.exit(1)

    return result

# ------------------------------------------------------------
# Seed operations
# ------------------------------------------------------------

def insert_test_tenant():
    phrase_hash = hash_passphrase(TEST_PASSPHRASE)

    sql = f"""
INSERT INTO tenants (id, name, access_phrase_hash, is_active)
VALUES (
    '{TEST_TENANT_ID}',
    '{TEST_TENANT_NAME}',
    '{phrase_hash}',
    TRUE
)
ON CONFLICT (access_phrase_hash) DO NOTHING
RETURNING id;
"""
    result = run_sql(sql, "INSERT test tenant")
    inserted_id = result.stdout.strip()

    if not inserted_id:
        print("[ERROR] Tenant insert affected 0 rows.")
        print("        A tenant with this passphrase hash already exists in the database.")
        print("        Either the database wasn't actually wiped before seeding, or this")
        print("        script has already been run once against it.")
        print(f"        Passphrase in use: {TEST_PASSPHRASE!r}")
        print("        Run this to see the existing tenant, then decide whether to reuse")
        print("        it or wipe the database (docker compose down -v) and re-seed:")
        print(f'        docker exec {CONTAINER_NAME} psql -U {PG_USER} -d {PG_DATABASE} '
              f'-c "SELECT id, name, created_at FROM tenants WHERE access_phrase_hash = '
              f"'{phrase_hash}';\"")
        sys.exit(1)

    if inserted_id != TEST_TENANT_ID:
        # Should be unreachable — RETURNING id can only return the row we
        # just inserted, using the id we supplied. Guarding anyway in case
        # a future change (trigger, different conflict target) breaks that
        # assumption silently.
        print("[ERROR] Insert returned an unexpected id — this should not happen.")
        print(f"        Expected: {TEST_TENANT_ID}")
        print(f"        Got:      {inserted_id}")
        sys.exit(1)

    print(f"[OK]    Tenant inserted.")
    print(f"        ID:   {inserted_id}")
    print(f"        Name: {TEST_TENANT_NAME}")
    print(f"        Hash: {phrase_hash}")


def verify_hash_lookup():
    phrase_hash = hash_passphrase(TEST_PASSPHRASE)

    sql = f"""
SELECT id, name
FROM tenants
WHERE access_phrase_hash = '{phrase_hash}'
  AND is_active = TRUE;
"""
    result = run_sql(sql, "Hash lookup verification")
    output = result.stdout.strip()

    if not output:
        print("[ERROR] Hash lookup returned no rows.")
        print("        The INSERT may have silently failed or the hash does not match.")
        sys.exit(1)

    parts         = output.split("|")
    returned_id   = parts[0].strip()
    returned_name = parts[1].strip()

    if returned_id != TEST_TENANT_ID:
        print(f"[ERROR] ID mismatch.")
        print(f"        Expected: {TEST_TENANT_ID}")
        print(f"        Got:      {returned_id}")
        sys.exit(1)

    print(f"[OK]    Hash lookup verified.")
    print(f"        Returned ID:   {returned_id}")
    print(f"        Returned name: {returned_name}")

# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main():
    print("=" * 60)
    print("  IDPP Seed Runner")
    print("=" * 60)

    print("[...] Inserting test tenant...")
    insert_test_tenant()

    print("[...] Verifying hash lookup...")
    verify_hash_lookup()

    print()
    print("=" * 60)
    print("  Seed complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()