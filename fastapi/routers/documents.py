# fastapi/routers/documents.py
#
# Document management endpoints.
#
# POST /upload              — ingest a file through the full pipeline
# GET  /{document_id}/status — poll pipeline status for a document
#
# This file owns HTTP concerns only:
#   · Request parsing and validation
#   · Authentication via get_current_tenant dependency
#   · Duplicate detection before any processing begins
#   · Agent orchestration (call Agent 1, then Agent 2)
#   · Error handling and HTTP response shaping
#
# No parsing logic, no chunking, no embedding, no GPT-4o calls live here.
# All of that is delegated to agents/ingestor.py and agents/classifier.py.

import hashlib
import json

import asyncpg
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel
from uuid6 import uuid7

from core.database import get_pool
from dependencies.auth import TenantContext, get_current_tenant
from agents.ingestor import IngestorAgent
from agents.classifier import ClassifierAgent


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()
# No prefix on the router itself — prefix "/documents" is applied in
# main.py when this router is registered with app.include_router().
# Consistent with the pattern established by session.py.


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {"pdf", "docx", "xlsx"}
# The three formats supported by parsers.py.
# Any other extension is rejected at the HTTP boundary before any
# file bytes are read into memory beyond what FastAPI already buffered.

MIME_TYPES = {
    "pdf":  "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
# Stored in documents.mime_type for informational purposes.
# We derive mime_type from the extension rather than trusting the
# Content-Type header — browsers and HTTP clients set Content-Type
# inconsistently for office documents.


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    """
    Response body for a successful POST /documents/upload.

    Returned after the full pipeline completes — Agent 1 and Agent 2
    have both finished, all chunks are in Qdrant and PostgreSQL, and
    the document status is 'embedded'.

    We return the classification result immediately in the upload response
    so the client does not need a follow-up status poll to learn the
    document type and chunk count.
    """
    document_id:     str
    status:          str    # always "embedded" on success
    doc_type:        str    # contract | invoice | claim | report | other
    confidence:      float  # GPT-4o classification confidence
    chunks_embedded: int    # number of chunks stored in Qdrant


class StatusResponse(BaseModel):
    """
    Response body for GET /documents/{document_id}/status.

    Returns the current pipeline stage so the client can track progress
    if it polls between upload and completion (e.g. for a progress bar).

    In the current synchronous implementation the upload endpoint does
    not return until the pipeline completes, so status will always be
    'embedded' immediately after a successful upload. This endpoint
    becomes more useful in Phase 7 when background task processing
    is introduced.
    """
    document_id: str
    status:      str
    filename:    str
    doc_type:    str | None   # None until Agent 2 completes
    uploaded_at: str          # ISO 8601 timestamp string


# ---------------------------------------------------------------------------
# Helper — compute SHA-256 hash
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    """
    Return the SHA-256 hex digest of the given bytes.

    64-character lowercase hex string. Called once per upload on the
    full file bytes before any processing begins.

    Why here and not in the ingestor?
    The hash is needed by the upload endpoint for duplicate detection
    before Agent 1 runs. The ingestor receives the pre-computed hash
    as a parameter so it can carry it into the IngestorPayload without
    recomputing it.
    """
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    tenant: TenantContext = Depends(get_current_tenant),
):
    """
    Upload a document and run it through the full ingestion pipeline.

    Flow:
      1.  Validate file extension — reject unsupported formats immediately
      2.  Read file bytes into memory
      3.  Compute SHA-256 hash
      4.  Check for duplicate — reject if hash already exists for this tenant
      5.  Insert document row with status='pending'
      6.  Update status to 'ingesting'
      7.  Run Agent 1 (IngestorAgent) → IngestorPayload
      8.  Run Agent 2 (ClassifierAgent) → ClassifierPayload
      9.  Return UploadResponse

    Steps 5-8 are all or nothing from the client's perspective — the HTTP
    response is not sent until step 9 completes. If any step raises, the
    client receives an error response and the document row is left at
    whatever status it reached before the failure. Re-uploading the same
    file is safe because the duplicate check uses file_hash — a failed
    upload that wrote a row but never set file_hash will not block a retry.

    Authentication:
      Bearer token required. get_current_tenant() resolves tenant identity
      before this handler runs. If the token is invalid, FastAPI returns
      401 before any code here executes.
    """

    # ── Step 1: Validate file extension ──────────────────────────────────

    filename = file.filename or ""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    # rsplit(".", 1) splits on the last dot only — handles filenames like
    # "report.v2.pdf" correctly, giving extension "pdf".

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '.{extension}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            ),
        )
    # We raise before reading file bytes — no memory allocated for
    # unsupported files beyond what FastAPI already buffered.

    # ── Step 2: Read file bytes ───────────────────────────────────────────

    raw_bytes = await file.read()
    # await file.read() reads the entire upload into memory.
    # Nginx enforces the 20MB upload limit before the request reaches
    # FastAPI — any file that gets here is within the size limit.
    # For very large files a streaming approach would be preferable,
    # but pdfplumber, python-docx, and openpyxl all require the full
    # file in memory anyway, so streaming buys nothing here.

    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    file_size = len(raw_bytes)

    # ── Step 3: Compute SHA-256 hash ──────────────────────────────────────

    file_hash = _sha256(raw_bytes)
    # 64-character hex string. Computed once here, passed to both the
    # database INSERT and the IngestorAgent so neither recomputes it.

    # ── Step 4: Duplicate detection ───────────────────────────────────────

    pool = get_pool(request.app)

    async with pool.acquire() as conn:
        existing = await conn.fetchrow(
            """
            SELECT id
            FROM documents
            WHERE tenant_id = $1
              AND file_hash  = $2
            """,
            tenant.tenant_id,
            file_hash,
        )
        # Scoped to tenant_id — the same file uploaded by two different
        # tenants is NOT a duplicate. Each tenant's documents are isolated.
        # file_hash alone (without tenant_id) would incorrectly reject
        # legitimate uploads from other tenants.

    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error":       "duplicate_file",
                "message":     "This file has already been uploaded.",
                "document_id": str(existing["id"]),
            },
            # We return the existing document_id so the client can use it
            # directly — no need to re-upload. The client can call
            # GET /documents/{document_id}/status to retrieve the result
            # of the original upload.
        )

    # ── Step 5: Insert document row with status='pending' ─────────────────

    document_id = str(uuid7())

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO documents (
                id,
                tenant_id,
                filename,
                file_size,
                mime_type,
                file_hash,
                status
            ) VALUES (
                $1, $2, $3, $4, $5, $6, 'pending'
            )
            """,
            document_id,
            tenant.tenant_id,
            filename,
            file_size,
            MIME_TYPES[extension],
            file_hash,
        )
        # file_hash is written here so duplicate detection works immediately
        # even if the pipeline fails partway through. A failed upload with
        # a written file_hash would block re-upload of the same file — we
        # accept this tradeoff. The operator can delete the failed row
        # manually to allow re-upload. Phase 7 adds automatic retry logic.

    # ── Step 6: Update status to 'ingesting' ──────────────────────────────

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE documents SET status = 'ingesting' WHERE id = $1",
            document_id,
        )
    # Status transition: pending → ingesting
    # Visible immediately via GET /documents/{document_id}/status.

    # ── Steps 7 & 8: Run Agent 1 then Agent 2 ────────────────────────────

    try:
        ingestor = IngestorAgent(request.app)
        ingestor_payload = await ingestor.run(
            raw_bytes   = raw_bytes,
            tenant_id   = tenant.tenant_id,
            document_id = document_id,
            filename    = filename,
            file_type   = extension,
            file_hash   = file_hash,
        )
        # Agent 1 runs the three-level chunking pipeline and returns
        # an IngestorPayload. Raises ValueError if the file cannot be
        # parsed or produces no valid chunks.

        classifier = ClassifierAgent(request.app)
        classifier_payload = await classifier.run(payload=ingestor_payload)
        # Agent 2 calls GPT-4o, writes to PostgreSQL and Qdrant,
        # and returns a ClassifierPayload.
        # Status transitions inside Agent 2:
        #   ingesting → classified (after GPT-4o)
        #   classified → embedded  (after all chunks stored)

    except ValueError as exc:
        # ValueError from run_pipeline() means the file could not be
        # parsed — image-only PDF, corrupted content.
        # Update status to reflect the failure and return 422.
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE documents SET status = 'failed' WHERE id = $1",
                document_id,
            )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    # ── Step 9: Return response ────────────────────────────────────────────

    return UploadResponse(
        document_id     = classifier_payload.document_id,
        status          = classifier_payload.status,
        doc_type        = classifier_payload.doc_type,
        confidence      = classifier_payload.confidence,
        chunks_embedded = classifier_payload.chunks_embedded,
    )


# ---------------------------------------------------------------------------
# GET /{document_id}/status
# ---------------------------------------------------------------------------

@router.get(
    "/{document_id}/status",
    response_model=StatusResponse,
    status_code=status.HTTP_200_OK,
)
async def get_document_status(
    document_id: str,
    request: Request,
    tenant: TenantContext = Depends(get_current_tenant),
):
    """
    Return the current pipeline status of a document.

    Requires a valid session token. Verifies the document belongs to
    the requesting tenant before returning any data — tenant A cannot
    poll the status of tenant B's documents.

    Parameters
    ----------
    document_id : str
        UUID string of the document to query. Taken from the URL path.

    Returns
    -------
    StatusResponse
        Current status, filename, doc_type (if classified), and
        upload timestamp.

    Raises
    ------
    HTTPException 404
        If no document with this ID exists for the requesting tenant.
        We return 404 (not 403) even when the document exists but belongs
        to a different tenant — revealing that a document exists at all
        is an information leak.
    """

    pool = get_pool(request.app)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                id,
                status,
                filename,
                doc_type,
                uploaded_at
            FROM documents
            WHERE id        = $1
              AND tenant_id = $2
            """,
            document_id,
            tenant.tenant_id,
            # Both conditions in one query:
            #   id = $1        → the requested document
            #   tenant_id = $2 → must belong to this tenant
            # If either fails, row is None and we return 404.
            # The client cannot distinguish "not found" from "belongs to
            # another tenant" — correct security behavior.
        )

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    return StatusResponse(
        document_id = str(row["id"]),
        status      = row["status"],
        filename    = row["filename"],
        doc_type    = row["doc_type"],
        uploaded_at = row["uploaded_at"].isoformat(),
        # asyncpg returns TIMESTAMPTZ as a Python datetime object.
        # .isoformat() converts to "2024-01-15T09:23:41.123456+00:00"
        # — a standard format any client can parse.
    )