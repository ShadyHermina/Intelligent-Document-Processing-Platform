# testing/MCP/test_search_documents.py
#
# Phase 8, Step 5 verification.
#
# Exercises the search_documents MCP tool through FastMCP's in-memory
# Client — the same protocol path FastAPI will use in Phase 9, but with an
# in-memory transport instead of TCP. The tool opens a real asyncpg
# connection to PostgreSQL, so this genuinely tests the DB path, not a mock.
#
# Run inside the fastmcp container:
#   docker cp testing/MCP/test_search_documents.py fastmcp:/tmp/test_search_documents.py
#   docker exec fastmcp python /tmp/test_search_documents.py
#
# FastMCP 2.3.4 result shape (confirmed by inspection):
#   client.call_tool(...) returns a list containing one TextContent object,
#   whose .text is a JSON string holding the tool's entire list[dict] return.
#   So we take result[0].text and json.loads it.
#
# Expected against current seed data (tenant bd8c8de3-...), 2 documents:
#   - unfiltered            -> 2 rows, test_document.pdf first (newest)
#   - doc_type="other"      -> 1 row  (test_document.pdf)
#   - status="failed"       -> 1 row  (GUC_Policy.pdf, doc_type is null)
#   - status="embedded"     -> 1 row  (test_document.pdf)

import asyncio
import json

from fastmcp import Client

import server  # the module under test; registers the tools and exposes mcp

SEED_TENANT = "bd8c8de3-4a8e-48b9-9065-9ac08918a9c7"


def _extract(result):
    """
    Unwrap a FastMCP 2.3.4 call_tool result into the tool's list[dict].

    Confirmed shape: result is a list of content blocks. The text block(s)
    carry the JSON-serialised return value. We concatenate any .text blocks
    and json.loads the result. An empty list (no content blocks) means the
    tool returned an empty result — represented here as [].
    """
    if isinstance(result, list):
        texts = [getattr(b, "text", None) for b in result]
        texts = [t for t in texts if t is not None]
        if not texts:
            return []
        return json.loads("".join(texts))
    # Fallback for any other shape (older/newer versions).
    for attr in ("data", "structured_content"):
        val = getattr(result, attr, None)
        if val is not None:
            if isinstance(val, dict) and "result" in val:
                return val["result"]
            return val
    return result


async def call(client, **kwargs):
    result = await client.call_tool("search_documents", kwargs)
    return _extract(result)


async def main():
    client = Client(server.mcp)
    async with client:
        print("=== unfiltered (expect 2 rows, newest first) ===")
        rows = await call(client, tenant_id=SEED_TENANT)
        for r in rows:
            print(f"  {r.get('filename'):20} status={r.get('status'):10} "
                  f"doc_type={r.get('doc_type')}")
        print(f"  COUNT = {len(rows)}\n")

        print("=== filter doc_type='other' (expect 1: test_document.pdf) ===")
        rows = await call(client, tenant_id=SEED_TENANT, doc_type="other")
        for r in rows:
            print(f"  {r.get('filename')}  doc_type={r.get('doc_type')}")
        print(f"  COUNT = {len(rows)}\n")

        print("=== filter status='failed' (expect 1: GUC_Policy.pdf) ===")
        rows = await call(client, tenant_id=SEED_TENANT, status="failed")
        for r in rows:
            print(f"  {r.get('filename')}  status={r.get('status')}  "
                  f"doc_type={r.get('doc_type')}")
        print(f"  COUNT = {len(rows)}\n")

        print("=== filter status='embedded' (expect 1: test_document.pdf) ===")
        rows = await call(client, tenant_id=SEED_TENANT, status="embedded")
        for r in rows:
            print(f"  {r.get('filename')}  status={r.get('status')}")
        print(f"  COUNT = {len(rows)}")


if __name__ == "__main__":
    asyncio.run(main())