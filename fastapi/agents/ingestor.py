# fastapi/agents/ingestor.py
#
# Agent 1 — Ingestor
#
# Responsibility: accept raw file bytes and produce a validated
# IngestorPayload containing all chunks and document-level metadata.
#
# What this file does NOT do:
#   · No database writes  — the upload endpoint handles those
#   · No OpenAI API calls — that is Agent 2's responsibility
#   · No parsing logic    — that lives in core/parsers.py
#
# The agent is a thin orchestrator. It reads configuration, calls
# run_pipeline(), and assembles the typed payload. Every heavy operation
# is delegated to parsers.py which is independently testable.

import hashlib
from fastapi import FastAPI

from core.config import get_settings
from core.parsers import run_pipeline
from agents.models import IngestorPayload


class IngestorAgent:
    """
    Agent 1: ingests raw file bytes and produces a structured IngestorPayload.

    Instantiated once per upload request in the documents router.
    Holds a reference to the FastAPI app instance to access:
      · app.state.st_model  — the SentenceTransformer loaded at startup
      · get_settings()      — for chunk_similarity_threshold

    Why instantiate per request rather than as a singleton?
    The agent holds app as an instance attribute. In tests, different
    test cases may use different app instances with different state.
    Per-request instantiation keeps each call fully isolated.
    The cost is negligible — __init__ does no I/O.
    """

    def __init__(self, app: FastAPI) -> None:
        self.app = app
        # Store the app reference. We do not read app.state here because
        # __init__ runs before we know the request is valid. We defer
        # reading app.state.st_model until run() is called, at which point
        # we know startup completed successfully.

    async def run(
        self,
        raw_bytes:   bytes,
        tenant_id:   str,
        document_id: str,
        filename:    str,
        file_type:   str,
        file_hash:   str,
    ) -> IngestorPayload:
        """
        Execute the ingestion pipeline on the uploaded file.

        Parameters
        ----------
        raw_bytes : bytes
            Raw file content as read from the multipart upload.
        tenant_id : str
            UUID string of the authenticated tenant.
        document_id : str
            UUID string assigned by the upload endpoint and already
            written to PostgreSQL before this method is called.
        filename : str
            Original filename from the upload — stored for audit purposes.
        file_type : str
            Lowercase extension without dot: "pdf", "docx", or "xlsx".
            Already validated by the upload endpoint.
        file_hash : str
            SHA-256 hex digest computed by the upload endpoint and already
            written to PostgreSQL. Passed here so the payload carries it
            without a database round-trip.

        Returns
        -------
        IngestorPayload
            Fully validated Pydantic model containing document metadata
            and the complete list of ChunkData objects ready for Agent 2.

        Raises
        ------
        ValueError
            If run_pipeline() cannot extract any text from the file,
            or if the file produces no valid chunks after all three levels.
            The upload endpoint catches this and returns HTTP 422.
        """

        settings = get_settings()
        # get_settings() returns the lru_cache singleton — zero cost after
        # the first call. We read it here (not in __init__) because tests
        # may patch settings between instantiation and the first run() call.

        st_model = self.app.state.st_model
        # app.state.st_model was set in main.py lifespan startup.
        # Reading it here — not importing sentence_transformers — keeps
        # this file independent of the heavy ML library.

        # ── Run the three-level chunking pipeline ────────────────────────

        chunks, location_count = run_pipeline(
            raw_bytes = raw_bytes,
            file_type = file_type,
            st_model  = st_model,
            threshold = settings.chunk_similarity_threshold,
        )
        # run_pipeline() applies:
        #   Level 1: format-specific structural parsing
        #   Pre-processing: normalize_text + boilerplate removal
        #   Level 2: sliding window semantic boundary detection
        #   Level 3: token size guardrails + sentence-level overlap
        #
        # Returns:
        #   chunks         — List[ChunkData], validated, indexed, token-counted
        #   location_count — int, total pages (PDF/DOCX) or sheets (XLSX)
        #
        # Raises ValueError if no text could be extracted or all text
        # was boilerplate. The upload endpoint handles this case.

        # ── Assemble and return the payload ─────────────────────────────

        return IngestorPayload(
            tenant_id      = tenant_id,
            document_id    = document_id,
            filename       = filename,
            file_type      = file_type,
            file_hash      = file_hash,
            location_count = location_count,
            chunks         = chunks,
        )
        # Pydantic validates every field at construction time:
        #   · file_hash must be exactly 64 characters (SHA-256 hex)
        #   · file_type must be one of "pdf", "docx", "xlsx"
        #   · chunks must be a non-empty list of valid ChunkData objects
        #   · location_count must be >= 1
        #
        # If any field fails validation, Pydantic raises ValidationError
        # which propagates to the upload endpoint as HTTP 500 — a signal
        # that the pipeline produced malformed output, not a client error.