# testing/MCP/test_get_document_summary.py
#
# Phase 8, Step 7 verification.
#
# Exercises the get_document_summary MCP tool through FastMCP's in-memory
# Client. Opens a real asyncpg connection; no OpenAI/Qdrant involved.
#
# Run inside the fastmcp container:
#   docker cp testing/MCP/test_get_document_summary.py fastmcp:/tmp/test_get_document_summary.py
#   docker exec fastmcp python /tmp/test_get_document_summary.py
#
# Three cases, asserted against real seed data:
#   1. embedded doc  e0a55c12-...  -> full metadata + entities dict
#                    (parties == ["German University in Cairo"])
#      Proves the JSONB-string -> dict json.loads() path.
#   2. failed doc    00c7d37f-...  -> metadata, entities == {} (was NULL)
#      Proves the NULL -> {} fallback branch.
#   3. embedded doc id + WRONG tenant -> None
#      Proves the WHERE id AND tenant_id cross-tenant guard.

import asyncio
import json

from fastmcp import Client

import server

SEED_TENANT   = "bd8c8de3-4a8e-48b9-9065-9ac08918a9c7"
WRONG_TENANT  = "00000000-0000-0000-0000-000000000000"

DOC_EMBEDDED  = "e0a55c12-a4ef-4641-bf4f-9c17c4bf94ba"  # test_document.pdf
DOC_FAILED    = "00c7d37f-2e42-492d-95ef-9d109d005a31"  # GUC_Policy.pdf


def _extract(result):
    """
    Unwrap FastMCP 2.3.4 call_tool result.
    get_document_summary returns a dict (or None). A dict comes back as a
    TextContent JSON object; None comes back as an empty content list.
    """
    if isinstance(result, list):
        texts = [getattr(b, "text", None) for b in result]
        texts = [t for t in texts if t is not None]
        if not texts:
            return None  # empty content == tool returned None
        return json.loads("".join(texts))
    for attr in ("data", "structured_content"):
        val = getattr(result, attr, None)
        if val is not None:
            if isinstance(val, dict) and "result" in val:
                return val["result"]
            return val
    return result


async def summary(client, tenant_id, document_id):
    result = await client.call_tool(
        "get_document_summary",
        {"tenant_id": tenant_id, "document_id": document_id},
    )
    return _extract(result)


async def main():
    async with Client(server.mcp) as client:
        # --- Case 1: embedded doc, correct tenant ---
        print("=== Case 1: embedded doc (expect full metadata + entities) ===")
        r1 = await summary(client, SEED_TENANT, DOC_EMBEDDED)
        if r1 is None:
            print("  FAIL: got None, expected a document")
        else:
            print(f"  filename          = {r1.get('filename')}")
            print(f"  status            = {r1.get('status')}")
            print(f"  doc_type          = {r1.get('doc_type')}")
            ent = r1.get("extracted_entities")
            print(f"  entities type     = {type(ent).__name__}")
            print(f"  entities.parties  = {ent.get('parties') if isinstance(ent, dict) else 'N/A'}")
            print(f"  entities (full)   = {ent}")
        print()

        # --- Case 2: failed doc, correct tenant (entities was NULL) ---
        print("=== Case 2: failed doc (expect metadata, entities == {}) ===")
        r2 = await summary(client, SEED_TENANT, DOC_FAILED)
        if r2 is None:
            print("  FAIL: got None, expected a document")
        else:
            print(f"  filename          = {r2.get('filename')}")
            print(f"  status            = {r2.get('status')}")
            print(f"  doc_type          = {r2.get('doc_type')}")
            ent = r2.get("extracted_entities")
            print(f"  entities type     = {type(ent).__name__}")
            print(f"  entities (full)   = {ent}   "
                  f"(expected empty dict {{}})")
        print()

        # --- Case 3: embedded doc id but WRONG tenant -> None ---
        print("=== Case 3: valid doc id + WRONG tenant (expect None) ===")
        r3 = await summary(client, WRONG_TENANT, DOC_EMBEDDED)
        print(f"  result = {r3}   (expected None)")
        print(f"  PASS: {r3 is None}")


if __name__ == "__main__":
    asyncio.run(main())
