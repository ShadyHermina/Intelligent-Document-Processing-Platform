# shared/embedding_service.py

import os
import uuid
from typing import List

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from shared.qdrant_store import get_qdrant_client

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
    qdrant_client: AsyncQdrantClient = await get_qdrant_client()

    vectors = await _embed_texts(openai_client, chunks)

    points: List[PointStruct] = []
    point_ids: List[str] = []

    for chunk_index, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
        point_id = str(uuid.uuid4())
        point_ids.append(point_id)

        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "tenant_id":   tenant_id,
                "document_id": document_id,
                "chunk_index": chunk_index,
                "text":        chunk_text,
            },
        )
        points.append(point)

    await qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    return point_ids


async def embed_and_query(
    query: str,
    tenant_id: str,
    top_k: int = 5,
) -> list:

    if not query:
        raise ValueError("query string cannot be empty")

    if not tenant_id:
        raise ValueError("tenant_id is required and cannot be empty")

    openai_client = get_openai_client()
    qdrant_client: AsyncQdrantClient = await get_qdrant_client()

    vectors = await _embed_texts(openai_client, [query])
    query_vector = vectors[0]

    results = await qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(value=tenant_id),
                )
            ]
        ),
        limit=top_k,
        with_payload=True,
    )

    return results