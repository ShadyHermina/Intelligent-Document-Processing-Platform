# testing/MCP/test_cross_tenant_tool_call.py
#
# Phase 8, Step 8 verification — tenant isolation across ALL three tools.
#
# For each tool, call it with the seed tenant (expect data) and with a
# foreign tenant UUID that owns nothing (expect empty — NOT an error, NOT
# another tenant's data). This proves isolation at both the PostgreSQL layer
# (search_documents, get_document_summary) and the Qdrant layer
# (query_knowledge_base's tenant payload filter).
#
# Run inside the fastmcp container:
#   docker cp testing/MCP/test_cross_tenant_tool_call.py fastmcp:/tmp/test_cross_tenant_tool_call.py
#   docker exec fastmcp python /tmp/test_cross_tenant_tool_call.py
#
# Note: query_knowledge_base makes a real OpenAI embedding call for BOTH the
# seed and foreign query. The foreign one still embeds, but the Qdrant filter
# returns nothing — which is the whole point.

import asyncio
import json

from fastmcp import Client

import server

SEED_TENANT    = "bd8c8de3-4a8e-48b9-9065-9ac08918a9c7"
FOREIGN_TENANT = "00000000-0000-0000-0000-000000000000"  # owns nothing
DOC_EMBEDDED   = "e0a55c12-a4ef-4641-bf4f-9c17c4bf94ba"


def _extract(result):
    """Unwrap FastMCP 2.3.4 result. Empty content list -> None (for dict-
    returning tools) which we normalise per-caller."""
    if isinstance(result, list):
        texts = [getattr(b, "text", None) for b in result]
        texts = [t for t in texts if t is not None]
        if not texts:
            return None
        return json.loads("".join(texts))
    for attr in ("data", "structured_content"):
        val = getattr(result, attr, None)
        if val is not None:
            if isinstance(val, dict) and "result" in val:
                return val["result"]
            return val
    return result


async def call(client, tool, args):
    result = await client.call_tool(tool, args)
    return _extract(result)


def report(tool, seed_result, foreign_result, seed_is_list):
    # Normalise: list tools use [] for empty; dict tool uses None.
    if seed_is_list:
        seed_len = len(seed_result) if seed_result else 0
        foreign_len = len(foreign_result) if foreign_result else 0
        seed_ok = seed_len > 0
        foreign_ok = foreign_len == 0
        print(f"[{tool}]")
        print(f"  seed    -> {seed_len} result(s)   (expect > 0)  {'OK' if seed_ok else 'FAIL'}")
        print(f"  foreign -> {foreign_len} result(s)   (expect 0)    {'OK' if foreign_ok else 'FAIL'}")
    else:
        seed_ok = seed_result is not None
        foreign_ok = foreign_result is None
        print(f"[{tool}]")
        print(f"  seed    -> {'a document' if seed_ok else 'None'}   (expect a document)  {'OK' if seed_ok else 'FAIL'}")
        print(f"  foreign -> {foreign_result}   (expect None)  {'OK' if foreign_ok else 'FAIL'}")
    print()
    return seed_ok and foreign_ok


async def main():
    all_pass = True
    async with Client(server.mcp) as client:
        # --- search_documents ---
        seed = await call(client, "search_documents", {"tenant_id": SEED_TENANT})
        foreign = await call(client, "search_documents", {"tenant_id": FOREIGN_TENANT})
        all_pass &= report("search_documents", seed, foreign, seed_is_list=True)

        # --- query_knowledge_base ---
        q = "attendance and academic integrity"
        seed = await call(client, "query_knowledge_base",
                          {"tenant_id": SEED_TENANT, "query": q})
        foreign = await call(client, "query_knowledge_base",
                             {"tenant_id": FOREIGN_TENANT, "query": q})
        all_pass &= report("query_knowledge_base", seed, foreign, seed_is_list=True)

        # --- get_document_summary ---
        seed = await call(client, "get_document_summary",
                          {"tenant_id": SEED_TENANT, "document_id": DOC_EMBEDDED})
        foreign = await call(client, "get_document_summary",
                             {"tenant_id": FOREIGN_TENANT, "document_id": DOC_EMBEDDED})
        all_pass &= report("get_document_summary", seed, foreign, seed_is_list=False)

        print("=" * 50)
        print(f"CROSS-TENANT ISOLATION: {'ALL PASS' if all_pass else 'FAILURE — investigate'}")


if __name__ == "__main__":
    asyncio.run(main())
