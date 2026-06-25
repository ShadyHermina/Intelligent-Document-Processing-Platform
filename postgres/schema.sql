-- postgres/schema.sql
-- Intelligent Document Processing Platform
-- Full schema definition: tenants, documents, chunks, audit_log
--
-- Run via: scripts/migrate.py
-- Do NOT execute manually unless testing teardown/rebuild.
--
-- Idempotency strategy: DROP IF EXISTS before every CREATE.
-- Running migrate.py twice is safe and produces the same result.

SET client_min_messages TO WARNING;

-- ============================================================
-- TEARDOWN — drop in reverse dependency order
-- ============================================================

DROP TABLE IF EXISTS chunks    CASCADE;
DROP TABLE IF EXISTS audit_log CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS tenants   CASCADE;

-- ============================================================
-- TABLE: tenants
-- ============================================================

CREATE TABLE tenants (
    id                 UUID        NOT NULL DEFAULT gen_random_uuid(),
    name               TEXT        NOT NULL,
    access_phrase_hash TEXT        NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active          BOOLEAN     NOT NULL DEFAULT TRUE,

    CONSTRAINT tenants_pkey
        PRIMARY KEY (id),

    CONSTRAINT tenants_access_phrase_hash_key
        UNIQUE (access_phrase_hash)
);

-- ============================================================
-- TABLE: documents
-- ============================================================

CREATE TABLE documents (
    id           UUID        NOT NULL DEFAULT gen_random_uuid(),
    tenant_id    UUID        NOT NULL,
    filename     TEXT        NOT NULL,
    file_size    BIGINT      NOT NULL,
    mime_type    TEXT        NOT NULL,
    status       TEXT        NOT NULL DEFAULT 'pending',
    uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ,

    CONSTRAINT documents_pkey
        PRIMARY KEY (id),

    CONSTRAINT documents_tenant_id_fkey
        FOREIGN KEY (tenant_id)
        REFERENCES tenants (id)
        ON DELETE CASCADE
);

CREATE INDEX idx_documents_tenant_id
    ON documents (tenant_id);

CREATE INDEX idx_documents_status
    ON documents (status);

CREATE INDEX idx_documents_uploaded_at
    ON documents (uploaded_at);

-- ============================================================
-- TABLE: chunks
-- ============================================================

CREATE TABLE chunks (
    id           UUID        NOT NULL DEFAULT gen_random_uuid(),
    document_id  UUID        NOT NULL,
    tenant_id    UUID        NOT NULL,
    chunk_index  INTEGER     NOT NULL,
    content      TEXT        NOT NULL,
    token_count  INTEGER     NOT NULL,
    embedding_id TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chunks_pkey
        PRIMARY KEY (id),

    CONSTRAINT chunks_document_id_fkey
        FOREIGN KEY (document_id)
        REFERENCES documents (id)
        ON DELETE CASCADE
);

CREATE INDEX idx_chunks_tenant_document
    ON chunks (tenant_id, document_id);

-- ============================================================
-- TABLE: audit_log
-- ============================================================

CREATE TABLE audit_log (
    id          BIGSERIAL   NOT NULL,
    tenant_id   UUID,
    actor       TEXT        NOT NULL,
    action      TEXT        NOT NULL,
    target_type TEXT,
    target_id   UUID,
    details     JSONB,
    timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT audit_log_pkey
        PRIMARY KEY (id),

    CONSTRAINT audit_log_tenant_id_fkey
        FOREIGN KEY (tenant_id)
        REFERENCES tenants (id)
        ON DELETE SET NULL
);

CREATE INDEX idx_audit_log_tenant_id
    ON audit_log (tenant_id);

CREATE INDEX idx_audit_log_timestamp
    ON audit_log (timestamp);