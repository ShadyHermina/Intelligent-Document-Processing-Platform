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
    chunks: List[str],
    tenant_id: str,
    document_id: str,
) -> List[str]:

    if not chunks:
        return []

    if not tenant_id:
        raise ValueError("tenant_id is required and cannot be empty")

    if not document_id:
        raise ValueError("document_id is required and cannot be empty")

    openai_client = get_openai_client()
    vectors = await _embed_texts(openai_client, chunks)

    point_ids: List[str] = []

    for chunk_index, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
        point_id = str(uuid.uuid4())
        point_ids.append(point_id)

        upsert_point(
            point_id  = point_id,
            vector    = vector,
            tenant_id = tenant_id,
            payload   = {
                "document_id": document_id,
                "chunk_index": chunk_index,
                "text":        chunk_text,
            },
        )

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