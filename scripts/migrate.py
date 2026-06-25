#!/usr/bin/env python3
"""
scripts/migrate.py
------------------
Applies postgres/schema.sql to the running PostgreSQL container.

Usage:
    python scripts/migrate.py

Expects:
    - Docker Desktop running
    - Container named 'postgres' is healthy
    - postgres/schema.sql exists relative to this script's parent directory
"""

import subprocess
import sys
import os

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

CONTAINER_NAME = "postgres"
PG_USER     = "idp_user"
PG_DATABASE = "idp_db"

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SCHEMA_PATH  = os.path.join(PROJECT_ROOT, "postgres", "schema.sql")

# ------------------------------------------------------------
# Pre-flight checks
# ------------------------------------------------------------

def check_schema_file_exists():
    if not os.path.isfile(SCHEMA_PATH):
        print(f"[ERROR] Schema file not found: {SCHEMA_PATH}")
        print("        Make sure postgres/schema.sql exists in the project root.")
        sys.exit(1)
    print(f"[OK]    Schema file found: {SCHEMA_PATH}")


def check_container_running():
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", CONTAINER_NAME],
        capture_output=True,
        text=True
    )
    if result.returncode != 0 or result.stdout.strip() != "true":
        print(f"[ERROR] Container '{CONTAINER_NAME}' is not running.")
        print("        Start it with: docker compose up -d")
        sys.exit(1)
    print(f"[OK]    Container '{CONTAINER_NAME}' is running.")

# ------------------------------------------------------------
# Migration runner
# ------------------------------------------------------------

def run_migration():
    print(f"[...] Reading schema from {SCHEMA_PATH}")

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    print(f"[OK]    Schema loaded ({len(schema_sql)} bytes).")
    print(f"[...] Applying schema to database '{PG_DATABASE}' as user '{PG_USER}'...")

    result = subprocess.run(
        [
            "docker", "exec",
            "-i",
            CONTAINER_NAME,
            "psql",
            "-U", PG_USER,
            "-d", PG_DATABASE,
            "-v", "ON_ERROR_STOP=1"
        ],
        input=schema_sql,
        capture_output=True,
        text=True
    )

    return result


def check_result(result):
    if result.stdout.strip():
        print("[psql stdout]")
        print(result.stdout)

    if result.stderr.strip():
        print("[psql stderr]")
        print(result.stderr)

    if result.returncode != 0:
        print(f"[ERROR] Migration failed with exit code {result.returncode}.")
        sys.exit(1)

    print("[OK]    Migration applied successfully.")

# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main():
    print("=" * 60)
    print("  IDP Migration Runner")
    print("=" * 60)

    check_schema_file_exists()
    check_container_running()

    result = run_migration()
    check_result(result)

    print()
    print("=" * 60)
    print("  Migration complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()