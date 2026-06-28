# test_batch_embedding.py
# Verifies that multiple chunks are embedded in a single OpenAI API call.
#
# Strategy:
#   We monkey-patch the OpenAI embeddings.create method to count how
#   many times it is called and capture the usage field from each call.
#   If batching works correctly, 12 chunks should produce exactly 1 API
#   call, and the usage.total_tokens should reflect all chunks combined.
#
# Run from inside the fastapi_a container:
#   python /tmp/test_batch_embedding.py

import asyncio
import os
from openai import AsyncOpenAI
from shared.embedding_service import _embed_texts

# ── Setup ─────────────────────────────────────────────────────────────────────

# 12 chunks — well above our assertion threshold of 10
# All about different aspects of Egyptian cities so token counts are meaningful
CHUNKS = [
    "Cairo is the largest city in Egypt and the Arab world.",
    "Alexandria is Egypt second largest city located on the Mediterranean coast.",
    "Giza is home to the famous pyramids and the Great Sphinx.",
    "Luxor contains some of the most important ancient Egyptian temples.",
    "Aswan is the southernmost city in Egypt situated along the Nile.",
    "Port Said is a major port city located at the northern end of the Suez Canal.",
    "Ismailia is a city on the west bank of the Suez Canal.",
    "Sharm el-Sheikh is a resort city on the southern tip of the Sinai Peninsula.",
    "Hurghada is a major tourist destination on the Red Sea coast.",
    "Mansoura is a city in the Nile Delta known for its university.",
    "Tanta is one of the largest cities in the Nile Delta region.",
    "Zagazig is the capital of the Sharqia Governorate in the Nile Delta.",
]

# ── Monkey-patch to count API calls ───────────────────────────────────────────

call_count = 0
call_details = []

async def main():
    global call_count, call_details

    api_key = os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key)

    # Wrap the real embeddings.create to intercept calls
    original_create = client.embeddings.create

    async def patched_create(**kwargs):
        global call_count, call_details
        call_count += 1
        response = await original_create(**kwargs)
        call_details.append({
            "call_number":  call_count,
            "input_count":  len(kwargs.get("input", [])),
            "total_tokens": response.usage.total_tokens,
        })
        return response

    client.embeddings.create = patched_create

    # ── Run the embedding ─────────────────────────────────────────
    print("Embedding " + str(len(CHUNKS)) + " chunks...")
    print("")

    vectors = await _embed_texts(client, CHUNKS)

    # ── Report results ────────────────────────────────────────────
    print("OpenAI API calls made: " + str(call_count))
    print("")
    for detail in call_details:
        print("  Call " + str(detail["call_number"]) + ":")
        print("    chunks sent    : " + str(detail["input_count"]))
        print("    total_tokens   : " + str(detail["total_tokens"]))
    print("")
    print("Vectors returned: " + str(len(vectors)))
    print("Vector dimension: " + str(len(vectors[0])))
    print("")

    # ── Assertions ────────────────────────────────────────────────

    # All 12 chunks fit in one batch (well under the 2048 limit)
    # so exactly 1 API call should have been made
    assert call_count == 1, (
        "Expected 1 API call for 12 chunks, got " + str(call_count)
    )

    # All 12 chunks were sent in that one call
    assert call_details[0]["input_count"] == len(CHUNKS), (
        "Expected " + str(len(CHUNKS)) + " chunks in call, "
        "got " + str(call_details[0]["input_count"])
    )

    # We got back exactly one vector per chunk
    assert len(vectors) == len(CHUNKS), (
        "Expected " + str(len(CHUNKS)) + " vectors, got " + str(len(vectors))
    )

    # Every vector has the correct dimension
    for i, vector in enumerate(vectors):
        assert len(vector) == 1536, (
            "Vector " + str(i) + " has wrong dimension: " + str(len(vector))
        )

    print("All assertions passed.")
    print("Batch embedding confirmed: " + str(len(CHUNKS))
          + " chunks embedded in " + str(call_count) + " API call.")


asyncio.run(main())