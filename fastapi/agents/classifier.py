# fastapi/agents/classifier.py
#
# Agent 2 — Classifier
#
# Responsibilities:
#   · Call GPT-4o to classify the document and extract structured entities
#   · Write classification results to PostgreSQL documents table
#   · Call embed_and_store() to embed all chunks and write to Qdrant
#   · Write all chunk rows to PostgreSQL chunks table
#   · Update document status through classified → embedded
#   · Write audit_log entry on pipeline completion
#   · Return ClassifierPayload to the upload endpoint
#
# What this file does NOT do:
#   · No file parsing — that is Agent 1's responsibility
#   · No chunking logic — that is parsers.py
#   · No direct Qdrant calls — that goes through embedding_service.py
#
# Database columns written by this agent:
#   documents.status             pending → classified → embedded
#   documents.doc_type           GPT-4o classification result
#   documents.extracted_entities GPT-4o entity extraction as JSONB
#   documents.processed_at       set when status reaches embedded
#   chunks.*                     one row per chunk, full extended payload
#   audit_log.*                  one row on pipeline completion

import json
from typing import List

from fastapi import FastAPI
from openai import AsyncOpenAI
from uuid6 import uuid7

from core.config import get_settings
from core.database import get_pool
from agents.models import IngestorPayload, ClassifierPayload
from shared.embedding_service import embed_and_store


# ---------------------------------------------------------------------------
# GPT-4o prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a document classifier for an enterprise document processing platform.

Your task is to analyze the provided document text and return a JSON object with this exact structure:
{
  "doc_type": "contract" | "invoice" | "claim" | "report" | "other",
  "confidence": <float between 0.0 and 1.0>,
  "entities": {
    "parties": ["list of organization or person names mentioned as parties"],
    "dates": {"effective": "YYYY-MM-DD or empty string", "expiry": "YYYY-MM-DD or empty string"},
    "amounts": ["list of monetary amounts as strings, e.g. '$50,000'"],
    "flags": ["list of notable clauses or risk indicators, e.g. 'automatic renewal clause'"]
  }
}

Rules:
- Return valid JSON only. No markdown. No explanation. No code fences.
- doc_type must be exactly one of the five values listed above.
- confidence must reflect how certain you are of the doc_type classification.
- If a field cannot be determined from the text, use an empty list [] or empty string "".
- Extract only what is explicitly present in the text. Do not infer or hallucinate."""

# Maximum number of tokens of document text to send to GPT-4o.
# We send the first N characters of concatenated chunk text.
# 12000 characters covers approximately 3000 tokens — enough for GPT-4o
# to classify the document type and extract entities from the opening
# sections where parties, dates, and amounts typically appear.
# Sending the entire document would be expensive and unnecessary for
# classification — the document type is almost always determinable
# from the first few pages.
_MAX_CLASSIFICATION_CHARS = 12_000


def _build_classification_text(payload: IngestorPayload) -> str:
    """
    Concatenate chunk texts up to _MAX_CLASSIFICATION_CHARS for GPT-4o.

    We take chunks in order (chunk_index order) and stop when we reach
    the character limit. This gives GPT-4o the opening of the document
    where document type signals are strongest.

    Parameters
    ----------
    payload : IngestorPayload
        The full ingestor output. Chunks are already in index order.

    Returns
    -------
    str
        Concatenated text, truncated at _MAX_CLASSIFICATION_CHARS.
    """
    parts: List[str] = []
    total_chars = 0

    for chunk in payload.chunks:
        remaining = _MAX_CLASSIFICATION_CHARS - total_chars
        if remaining <= 0:
            break
        # Take as much of this chunk as fits within the limit.
        parts.append(chunk.text[:remaining])
        total_chars += min(len(chunk.text), remaining)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# ClassifierAgent
# ---------------------------------------------------------------------------

class ClassifierAgent:
    """
    Agent 2: classifies an ingested document and writes all results to
    PostgreSQL and Qdrant.

    Instantiated once per upload request in the documents router.
    Holds a reference to the FastAPI app instance to access the
    database pool via get_pool(app).

    Why per-request instantiation and not a singleton?
    Same reasoning as IngestorAgent: __init__ does no I/O, construction
    is nanoseconds, and per-request isolation keeps tests clean.
    See ingestor.py for the full explanation.
    """

    def __init__(self, app: FastAPI) -> None:
        self.app = app

    async def run(self, payload: IngestorPayload) -> ClassifierPayload:
        """
        Execute the classification and embedding pipeline.

        Parameters
        ----------
        payload : IngestorPayload
            Complete output of Agent 1. Contains document metadata and
            the full list of ChunkData objects ready for embedding.

        Returns
        -------
        ClassifierPayload
            Classification result and pipeline completion summary.
            Returned to the upload endpoint to build the HTTP response.

        Raises
        ------
        Exception
            Any unhandled exception from OpenAI, asyncpg, or Qdrant
            propagates to the upload endpoint which returns HTTP 500.
            Document status may be left at "classified" if failure
            occurs after the GPT-4o call but before the embedding loop.
            This is visible in the database and recoverable by re-upload
            (duplicate detection uses file_hash, not document_id).
        """

        settings = get_settings()
        pool = get_pool(self.app)
        # get_pool() is synchronous — retrieves app.state.db_pool.
        # All database operations in this method use this pool.

        # ── Step 1: Call GPT-4o for classification and entity extraction ──

        classification_text = _build_classification_text(payload)

        openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        # We instantiate AsyncOpenAI directly here using the typed settings
        # value rather than os.getenv() — consistent with the rest of the
        # codebase and validated at startup via pydantic-settings.

        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            # response_format json_object instructs GPT-4o to return valid
            # JSON only. Combined with the system prompt instruction, this
            # makes JSON parsing failures extremely rare.
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": classification_text},
            ],
            temperature=0.1,
            # temperature=0.1: near-deterministic output for classification.
            # We want consistent results for the same document, not creative
            # variation. 0.0 is fully deterministic but occasionally causes
            # GPT-4o to get stuck in repetition loops — 0.1 avoids this.
            max_tokens=1000,
            # 1000 tokens is more than enough for the JSON response structure.
            # A full entity extraction with many parties, dates, and flags
            # rarely exceeds 400 tokens.
        )

        raw_json = response.choices[0].message.content or "{}"

        try:
            gpt_result = json.loads(raw_json)
        except json.JSONDecodeError:
            # GPT-4o returned something that is not valid JSON despite
            # response_format=json_object. Rare but possible if the model
            # truncated mid-token at max_tokens. Fall back to safe defaults.
            gpt_result = {}

        # Extract and validate each field from GPT-4o response.
        # We do not trust GPT-4o to return exactly the values we asked for
        # — we validate and sanitize every field before writing to the DB.

        VALID_DOC_TYPES = {"contract", "invoice", "claim", "report", "other"}

        raw_doc_type = str(gpt_result.get("doc_type", "other")).lower().strip()
        doc_type = raw_doc_type if raw_doc_type in VALID_DOC_TYPES else "other"
        # If GPT-4o returns "Contract" (capitalized) or "purchase_order"
        # (not in our set), we fall back to "other" rather than crashing
        # or storing an invalid value in the database.

        raw_confidence = gpt_result.get("confidence", 0.0)
        try:
            confidence = float(raw_confidence)
            confidence = max(0.0, min(1.0, confidence))
            # Clamp to [0.0, 1.0] in case GPT-4o returns 1.2 or -0.1.
        except (TypeError, ValueError):
            confidence = 0.0

        raw_entities = gpt_result.get("entities", {})
        entities = raw_entities if isinstance(raw_entities, dict) else {}
        # Ensure entities is always a dict — ClassifierPayload.entities
        # is typed as Dict[str, Any] and Pydantic will reject non-dicts.

        # ── Step 2: Write classification results to PostgreSQL ────────────

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE documents
                SET
                    status             = 'classified',
                    doc_type           = $1,
                    extracted_entities = $2
                WHERE id = $3
                """,
                doc_type,
                json.dumps(entities),
                # asyncpg accepts JSONB as a JSON string — json.dumps()
                # converts the dict to a string that PostgreSQL casts to JSONB.
                payload.document_id,
            )
            # Status transitions:
            #   pending → ingesting (upload endpoint, before Agent 1)
            #   ingesting → classified (here, after GPT-4o)
            #   classified → embedded (below, after all chunks stored)

        # ── Step 3: Embed all chunks and store in Qdrant ──────────────────

        point_ids = await embed_and_store(
            chunks      = payload.chunks,
            tenant_id   = payload.tenant_id,
            document_id = payload.document_id,
            doc_type    = doc_type,
            file_type   = payload.file_type,
        )
        # embed_and_store() calls OpenAI text-embedding-3-small once for
        # all chunks in one batched API call, then upserts each vector
        # into Qdrant with the full extended payload.
        #
        # point_ids is a List[str] in the same order as payload.chunks.
        # We zip them together below when writing PostgreSQL chunk rows.

        # ── Step 4: Write chunk rows to PostgreSQL ────────────────────────

        async with pool.acquire() as conn:
            for chunk, point_id in zip(payload.chunks, point_ids):
                await conn.execute(
                    """
                    INSERT INTO chunks (
                        id,
                        document_id,
                        tenant_id,
                        chunk_index,
                        content,
                        token_count,
                        embedding_id,
                        location_index,
                        section_label,
                        image_present,
                        file_type,
                        doc_type
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
                    )
                    """,
                    str(uuid7()),  # $1  id — new UUID for PostgreSQL PK
                    payload.document_id,# $2  document_id
                    payload.tenant_id,  # $3  tenant_id
                    chunk.chunk_index,  # $4  chunk_index
                    chunk.text,         # $5  content (column is named content)
                    chunk.token_count,  # $6  token_count
                    point_id,           # $7  embedding_id — Qdrant point_id
                    chunk.location_index,# $8 location_index
                    chunk.section_label, # $9 section_label
                    chunk.image_present, # $10 image_present
                    payload.file_type,  # $11 file_type
                    doc_type,           # $12 doc_type
                )
                # Each chunk row links PostgreSQL and Qdrant via embedding_id.
                # Given a chunk row in PostgreSQL, embedding_id retrieves the
                # corresponding Qdrant point for the vector.
                # Given a Qdrant search result, the point_id matches
                # embedding_id in PostgreSQL for the full relational context.

        # ── Step 5: Update document status to embedded ────────────────────

        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE documents
                SET
                    status       = 'embedded',
                    processed_at = now()
                WHERE id = $1
                """,
                payload.document_id,
            )
            # processed_at records when the full pipeline completed.
            # This column already exists in the Phase 1 schema.
            # Status is now at its terminal value: embedded.

        # ── Step 6: Write audit log entry ────────────────────────────────

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO audit_log (
                    tenant_id,
                    actor,
                    action,
                    target_type,
                    target_id,
                    details
                ) VALUES (
                    $1, $2, $3, $4, $5, $6
                )
                """,
                payload.tenant_id,          # $1 tenant_id
                payload.tenant_id,          # $2 actor — the tenant performed this action
                "pipeline_complete",        # $3 action
                "document",                 # $4 target_type
                payload.document_id,        # $5 target_id
                json.dumps({                # $6 details — JSONB
                    "chunks_embedded": len(point_ids),
                    "doc_type":        doc_type,
                    "confidence":      confidence,
                    "file_type":       payload.file_type,
                    "filename":        payload.filename,
                }),
                # audit_log.details is JSONB — we store a summary of the
                # pipeline result so operators can review what was processed
                # without querying the documents or chunks tables.
                # actor = tenant_id: the authenticated tenant triggered this.
                # In a multi-user tenant model this would be a user_id.
            )

        # ── Step 7: Return ClassifierPayload ─────────────────────────────

        return ClassifierPayload(
            tenant_id       = payload.tenant_id,
            document_id     = payload.document_id,
            doc_type        = doc_type,
            confidence      = confidence,
            entities        = entities,
            chunks_embedded = len(point_ids),
            status          = "embedded",
        )