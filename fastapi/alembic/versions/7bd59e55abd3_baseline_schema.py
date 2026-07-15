"""baseline schema

Captures the FULL current schema as of Phase 10 — the merged result of
three previously separate, uncoordinated sources:
    postgres/schema.sql          (tenants, documents, chunks, audit_log)
    scripts/phase6_migrate.py    (documents/chunks columns added in Phase 6)
    scripts/add_sessions_table.py (sessions table)

This migration is the new single source of truth for schema state going
forward. The three scripts above are kept in the repo for historical
reference only — do not run them against a database managed by Alembic.

IMPORTANT — existing databases:
    If you're running this against a database that already has these
    tables (i.e. any existing dev/test environment), do NOT run
    `alembic upgrade head` — it will fail on CREATE TABLE for tables
    that already exist. Instead run `alembic stamp head` once, which
    tells Alembic "this database is already at this state" without
    executing any SQL. See the chat instructions for the exact command.

Fresh databases (new clone, CI, a new teammate's machine) should run
`alembic upgrade head` normally — this migration builds the entire
schema from nothing.

Revision ID: 7bd59e55abd3
Revises:
Create Date: 2026-07-15 09:37:47.560886

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7bd59e55abd3'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ---------------------------------------------------------------------------
# We use raw SQL via op.execute() rather than SQLAlchemy's op.create_table()
# Column DSL. This project has no ORM layer anywhere — every query in the
# application is raw SQL via asyncpg. Writing this baseline as raw SQL keeps
# it byte-for-byte identical to what schema.sql/phase6_migrate.py/
# add_sessions_table.py actually produced, rather than introducing a second,
# slightly different dialect (SQLAlchemy Column types) for describing the
# same tables. Future migrations should follow the same raw-SQL convention.
# ---------------------------------------------------------------------------


def upgrade() -> None:
    # ── tenants ────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE tenants (
            id                 UUID        NOT NULL DEFAULT gen_random_uuid(),
            name               TEXT        NOT NULL,
            access_phrase_hash TEXT        NOT NULL,
            created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            is_active          BOOLEAN     NOT NULL DEFAULT TRUE,

            CONSTRAINT tenants_pkey PRIMARY KEY (id),
            CONSTRAINT tenants_access_phrase_hash_key UNIQUE (access_phrase_hash)
        );
    """)

    # ── documents (base columns + Phase 6 additions merged) ────────────
    op.execute("""
        CREATE TABLE documents (
            id                 UUID        NOT NULL DEFAULT gen_random_uuid(),
            tenant_id          UUID        NOT NULL,
            filename           TEXT        NOT NULL,
            file_size          BIGINT      NOT NULL,
            mime_type          TEXT        NOT NULL,
            status             TEXT        NOT NULL DEFAULT 'pending',
            uploaded_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            processed_at       TIMESTAMPTZ,
            file_hash          TEXT,
            doc_type           TEXT,
            extracted_entities JSONB,

            CONSTRAINT documents_pkey PRIMARY KEY (id),
            CONSTRAINT documents_tenant_id_fkey
                FOREIGN KEY (tenant_id) REFERENCES tenants (id) ON DELETE CASCADE
        );
    """)
    op.execute("CREATE INDEX idx_documents_tenant_id   ON documents (tenant_id);")
    op.execute("CREATE INDEX idx_documents_status      ON documents (status);")
    op.execute("CREATE INDEX idx_documents_uploaded_at ON documents (uploaded_at);")

    # ── chunks (base columns + Phase 6 additions merged) ────────────────
    op.execute("""
        CREATE TABLE chunks (
            id             UUID        NOT NULL DEFAULT gen_random_uuid(),
            document_id    UUID        NOT NULL,
            tenant_id      UUID        NOT NULL,
            chunk_index    INTEGER     NOT NULL,
            content        TEXT        NOT NULL,
            token_count    INTEGER     NOT NULL,
            embedding_id   TEXT,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            location_index INTEGER,
            section_label  TEXT,
            image_present  BOOLEAN,
            file_type      TEXT,
            doc_type       TEXT,

            CONSTRAINT chunks_pkey PRIMARY KEY (id),
            CONSTRAINT chunks_document_id_fkey
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
        );
    """)
    op.execute("CREATE INDEX idx_chunks_tenant_document ON chunks (tenant_id, document_id);")

    # ── sessions (from scripts/add_sessions_table.py) ───────────────────
    op.execute("""
        CREATE TABLE sessions (
            token      TEXT        PRIMARY KEY,
            tenant_id  UUID        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            expires_at TIMESTAMPTZ NOT NULL
        );
    """)
    op.execute("CREATE INDEX idx_sessions_expires_at ON sessions(expires_at);")

    # ── audit_log ────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE audit_log (
            id          BIGSERIAL   NOT NULL,
            tenant_id   UUID,
            actor       TEXT        NOT NULL,
            action      TEXT        NOT NULL,
            target_type TEXT,
            target_id   UUID,
            details     JSONB,
            timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

            CONSTRAINT audit_log_pkey PRIMARY KEY (id),
            CONSTRAINT audit_log_tenant_id_fkey
                FOREIGN KEY (tenant_id) REFERENCES tenants (id) ON DELETE SET NULL
        );
    """)
    op.execute("CREATE INDEX idx_audit_log_tenant_id ON audit_log (tenant_id);")
    op.execute("CREATE INDEX idx_audit_log_timestamp ON audit_log (timestamp);")


def downgrade() -> None:
    # Reverse dependency order — same order schema.sql's own teardown used.
    op.execute("DROP TABLE IF EXISTS sessions   CASCADE;")
    op.execute("DROP TABLE IF EXISTS chunks     CASCADE;")
    op.execute("DROP TABLE IF EXISTS audit_log  CASCADE;")
    op.execute("DROP TABLE IF EXISTS documents  CASCADE;")
    op.execute("DROP TABLE IF EXISTS tenants    CASCADE;")
