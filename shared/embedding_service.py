# shared/embedding_service.py

import os
import uuid
from typing import List

from openai import AsyncOpenAI

from shared.qdrant_store import get_client, upsert_point, search_with_tenant

EMBEDDING_MODEL      = "text-embedding-3-small"
COLLECTION_NAME      = "document_chunks"
OPENAI_BATCH_SIZE    = 2048
EMBEDDING_DIMENSIONS = 1536


def get_openai_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    return AsyncOpenAI(api_key=api_key)


async def _embed_texts(
    client: AsyncOpenAI,
    texts: List[str],
) -> List[List[float]]:

    all_embeddings: List[List[float]] = []

    for batch_start in range(0, len(texts), OPENAI_BATCH_SIZE):
        batch = texts[batch_start : batch_start + OPENAI_BATCH_SIZE]

        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


async def embed_and_store(
    chunks:      List,    # List[ChunkData] — typed as List to avoid
    tenant_id:   str,     # importing ChunkData here and creating a
    document_id: str,     # circular import between shared/ and agents/
    doc_type:    str,
    file_type:   str,
) -> List[str]:
    """
    Embed a list of ChunkData objects and store each vector in Qdrant.

    Accepts the full list of chunks and embeds all texts in one batched
    OpenAI API call via _embed_texts(). Then upserts each vector into
    Qdrant with the full extended payload carrying all metadata needed
    for downstream retrieval and filtering.

    Parameters
    ----------
    chunks : List[ChunkData]
        All chunks from IngestorPayload. Each carries text, chunk_index,
        location_index, section_label, image_present, token_count.
        Typed as List (not List[ChunkData]) to avoid importing from
        agents/ into shared/ — shared/ must not depend on fastapi/.
    tenant_id : str
        UUID string of the tenant. Added to every Qdrant payload by
        upsert_point() automatically, enforcing tenant isolation.
    document_id : str
        UUID string of the document these chunks belong to.
    doc_type : str
        Classification result from Agent 2 GPT-4o call.
        e.g. "contract", "invoice", "claim", "report", "other"
    file_type : str
        Source format: "pdf", "docx", or "xlsx".

    Returns
    -------
    List[str]
        Qdrant point_ids in the same order as the input chunks.
        Agent 2 uses these as embedding_id when writing PostgreSQL
        chunks rows — linking each PostgreSQL chunk to its Qdrant point.
    """

    if not chunks:
        return []

    if not tenant_id:
        raise ValueError("tenant_id is required and cannot be empty")

    if not document_id:
        raise ValueError("document_id is required and cannot be empty")

    # Extract text from each ChunkData for the batched embedding call.
    # All texts go to OpenAI in one request — one API call regardless
    # of how many chunks the document produced.
    texts = [chunk.text for chunk in chunks]

    openai_client = get_openai_client()
    vectors = await _embed_texts(openai_client, texts)
    # vectors is a List[List[float]] in the same order as texts.
    # zip(chunks, vectors) pairs each ChunkData with its embedding vector.

    point_ids: List[str] = []

    for chunk, vector in zip(chunks, vectors):
        point_id = str(uuid.uuid4())
        point_ids.append(point_id)

        upsert_point(
            point_id  = point_id,
            vector    = vector,
            tenant_id = tenant_id,
            # tenant_id is also merged into the payload inside upsert_point()
            # automatically — we do not need to add it here manually.
            payload   = {
                "document_id":    document_id,
                "chunk_index":    chunk.chunk_index,
                "text":           chunk.text,
                "location_index": chunk.location_index,
                "section_label":  chunk.section_label,
                "image_present":  chunk.image_present,
                "doc_type":       doc_type,
                "file_type":      file_type,
            },
        )
        # Every Qdrant point now carries the full context needed for
        # retrieval queries:
        #   tenant_id      → isolation filter (every search uses this)
        #   document_id    → "find all chunks for this document"
        #   chunk_index    → reconstruct document order
        #   text           → return the actual text to the LLM
        #   location_index → "find chunks from page 3" or "sheet 2"
        #   section_label  → "find chunks under Indemnification clause"
        #   image_present  → "find chunks with figures"
        #   doc_type       → "find all contract chunks for this tenant"
        #   file_type      → "find all PDF chunks for this tenant"

    return point_ids


async def embed_and_query(
    query: str,
    tenant_id: str,
    top_k: int = 5,
) -> List[dict]:

    if not query:
        raise ValueError("query string cannot be empty")

    if not tenant_id:
        raise ValueError("tenant_id is required and cannot be empty")

    openai_client = get_openai_client()

    vectors = await _embed_texts(openai_client, [query])
    query_vector = vectors[0]

    results = search_with_tenant(
        query_vector = query_vector,
        tenant_id    = tenant_id,
        limit        = top_k,
    )

    return results