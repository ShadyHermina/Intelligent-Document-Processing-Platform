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
import uuid
import hashlib

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

CONTAINER_NAME = "postgres"
PG_USER        = "idp_user"
PG_DATABASE    = "idp_db"

TEST_TENANT_NAME   = "IDPP Test Tenant"
TEST_PASSPHRASE    = "correct-horse-battery-staple"
TEST_TENANT_ID     = str(uuid.uuid4())

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
ON CONFLICT (access_phrase_hash) DO NOTHING;
"""
    run_sql(sql, "INSERT test tenant")
    print(f"[OK]    Tenant inserted.")
    print(f"        ID:   {TEST_TENANT_ID}")
    print(f"        Name: {TEST_TENANT_NAME}")
    print(f"        Hash: {hash_passphrase(TEST_PASSPHRASE)}")


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