# fastapi/agents/models.py
#
# Pydantic models that define the data contracts between pipeline stages.
#
# Three models, three boundaries:
#
#   ChunkData        — one processed chunk produced by Agent 1
#   IngestorPayload  — Agent 1 → Agent 2  (full document + all chunks)
#   ClassifierPayload— Agent 2 → upload endpoint (classification result)
#
# Why Pydantic models and not plain dataclasses or dicts?
#
# Pydantic validates every field at construction time. If Agent 1 tries
# to build a ChunkData with chunk_index="zero" instead of chunk_index=0,
# Pydantic raises a ValidationError immediately at the source — not a
# cryptic TypeError three function calls later inside Agent 2. This makes
# bugs visible at the boundary where they are introduced, not at the point
# where they eventually cause a failure.
#
# Plain dicts have no validation and no type information. A typo in a key
# name ("chunkindex" instead of "chunk_index") silently passes through
# until something tries to read the missing key.
#
# Dataclasses give typed attributes but no validation — the wrong type
# is stored silently.
#
# Pydantic gives both typed attributes AND validation at construction time.

from typing import Dict, List, Literal, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# ChunkData
# ---------------------------------------------------------------------------

class ChunkData(BaseModel):
    """
    One processed chunk output by Agent 1's three-level chunking pipeline.

    Carries both the chunk text and all positional/structural metadata
    so Agent 2 can write the full extended payload to Qdrant and PostgreSQL
    without any additional lookups.

    Every ChunkData instance is immutable after construction — Pydantic
    models are not frozen by default, but we treat them as read-only.
    Agent 2 reads fields; it never modifies them.
    """

    chunk_index: int = Field(
        ...,
        ge=0,
        description="Zero-based position of this chunk within the document.",
    )
    # ge=0 means greater-than-or-equal-to zero. chunk_index=-1 raises
    # ValidationError immediately. The ... means the field is required —
    # no default value. Pydantic raises if it is absent.

    text: str = Field(
        ...,
        min_length=1,
        description="Normalized chunk text after all three pipeline levels.",
    )
    # min_length=1 prevents empty-string chunks from being stored.
    # An empty chunk would produce a meaningless embedding vector and
    # a useless PostgreSQL row.

    location_index: int = Field(
        ...,
        ge=1,
        description=(
            "Page number (PDF/DOCX, 1-based) or sheet index (XLSX, 1-based). "
            "Format-agnostic positional locator."
        ),
    )
    # ge=1 because pages and sheets are 1-based in every format we support.
    # A location_index of 0 would indicate a bug in the parser.

    section_label: str = Field(
        ...,
        min_length=1,
        description=(
            "Structural label for the region this chunk came from. "
            "PDF/DOCX: nearest heading text, or 'Body' if no heading. "
            "XLSX: '{sheet_name} / row group {N}'."
        ),
    )
    # min_length=1 ensures parsers always supply a label.
    # "Body" is the canonical fallback — never an empty string.

    image_present: bool = Field(
        default=False,
        description=(
            "True if an image was detected adjacent to this chunk's text. "
            "No multimodal embedding is performed — detection flag only."
        ),
    )
    # Default False: the common case is no image. Parsers set True only
    # when they detect an image bounding box within 50px (PDF) or an
    # image paragraph immediately preceding this chunk's text (DOCX).

    token_count: int = Field(
        ...,
        ge=1,
        description=(
            "Token count using cl100k_base encoding (tiktoken). "
            "Guaranteed between 50 and 400 after Level 3 guardrails."
        ),
    )
    # ge=1 because a chunk with zero tokens cannot exist after min_length=1
    # on text. The 50-400 range is enforced by the chunking pipeline;
    # we do not re-enforce it here to avoid double validation.


# ---------------------------------------------------------------------------
# IngestorPayload
# ---------------------------------------------------------------------------

class IngestorPayload(BaseModel):
    """
    Complete output of Agent 1. Passed directly to Agent 2.

    Carries document-level metadata and the full list of ChunkData objects
    produced by the three-level chunking pipeline. Agent 2 reads everything
    it needs from this model — no further file reading, no re-parsing.
    """

    tenant_id: str = Field(
        ...,
        description="UUID string of the tenant who uploaded this document.",
    )
    # str rather than uuid.UUID because all layers in this application
    # work with string UUIDs — consistent with TenantContext, asyncpg
    # row access, and Qdrant payload values.

    document_id: str = Field(
        ...,
        description="UUID string assigned to this document by the upload endpoint.",
    )

    filename: str = Field(
        ...,
        min_length=1,
        description="Original filename as uploaded by the client.",
    )

    file_type: Literal["pdf", "docx", "xlsx"] = Field(
        ...,
        description="Detected file format. Exactly one of three allowed values.",
    )
    # Literal["pdf", "docx", "xlsx"] means Pydantic raises ValidationError
    # if any other string is supplied. This is tighter than str — it
    # documents the allowed values AND enforces them at construction time.
    # Alternative: an Enum. Literal is simpler for a fixed three-value set
    # that will not grow without a code change anyway.

    file_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hex digest of the raw file bytes (64 hex characters).",
    )
    # min_length=max_length=64 enforces the exact length of a SHA-256
    # hex string. A 63-character value indicates a truncation bug.
    # A 65-character value indicates a formatting bug. Both fail loudly.

    location_count: int = Field(
        ...,
        ge=1,
        description=(
            "Total number of structural location units: "
            "page count for PDF/DOCX, sheet count for XLSX."
        ),
    )
    # ge=1 because a document with zero pages or zero sheets is not a
    # valid document. Parser would have raised before reaching this point,
    # but we validate here as a second layer of defence.

    chunks: List[ChunkData] = Field(
        ...,
        min_length=1,
        description="Ordered list of all chunks produced by the pipeline.",
    )
    # min_length=1 means a document that produced zero chunks fails
    # validation here rather than silently producing an empty Qdrant
    # collection and a document row with status='embedded' but no data.


# ---------------------------------------------------------------------------
# ClassifierPayload
# ---------------------------------------------------------------------------

class ClassifierPayload(BaseModel):
    """
    Complete output of Agent 2. Returned to the upload endpoint.

    Carries the classification result and a summary of what was written
    to PostgreSQL and Qdrant. The upload endpoint uses this to build
    the HTTP response returned to the client.
    """

    tenant_id: str = Field(
        ...,
        description="UUID string of the tenant — passed through from IngestorPayload.",
    )

    document_id: str = Field(
        ...,
        description="UUID string of the document — passed through from IngestorPayload.",
    )

    doc_type: Literal["contract", "invoice", "claim", "report", "other"] = Field(
        ...,
        description="Document classification assigned by GPT-4o.",
    )
    # Same Literal pattern as file_type — enforces the allowed values
    # that GPT-4o is instructed to return. If GPT-4o returns something
    # outside these values, Agent 2 maps it to "other" before constructing
    # this model, so this validation should always pass in normal operation.

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="GPT-4o confidence score for the doc_type classification.",
    )
    # ge=0.0, le=1.0 constrains to the probability range.
    # A confidence of 1.2 would indicate a GPT-4o response parsing bug.

    entities: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured entities extracted by GPT-4o. "
            "Keys: parties (list), dates (dict), amounts (list), flags (list). "
            "Stored as JSONB in documents.extracted_entities."
        ),
    )
    # Dict[str, Any] rather than a nested Pydantic model because the
    # entity structure may be partially populated — GPT-4o may not find
    # parties in a financial report, or dates in a claim form.
    # A nested model with required fields would reject valid partial results.
    # default_factory=dict means an empty dict is valid when GPT-4o
    # extracts nothing — preferable to a ValidationError on empty documents.

    chunks_embedded: int = Field(
        ...,
        ge=0,
        description="Number of chunks successfully written to Qdrant and PostgreSQL.",
    )
    # ge=0 rather than ge=1 because a theoretical edge case of zero
    # embedded chunks (all chunks rejected by guardrails) should surface
    # as a data anomaly to investigate, not a validation failure that
    # masks the root cause.

    status: Literal["embedded"] = Field(
        default="embedded",
        description=(
            "Terminal pipeline status. Always 'embedded' when this payload "
            "is constructed — Agent 2 only returns after all writes complete."
        ),
    )
    # Literal["embedded"] with a default means:
    #   · The field is always present in the model
    #   · Its value is always exactly "embedded"
    #   · No other value is accepted
    # This is stricter than status: str = "embedded" which would accept
    # any string if someone explicitly passed a different value.