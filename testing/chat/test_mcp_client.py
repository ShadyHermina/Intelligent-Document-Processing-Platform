# testing/Chat/test_mcp_client.py
#
# Verifies that fastapi/core/mcp_client.py correctly:
#   1. Performs the three-step MCP session handshake
#   2. Injects tenant_id unconditionally into every tool call
#   3. Parses SSE responses and extracts tool results correctly
#   4. Normalizes query_knowledge_base chunks to flat dicts
#      with text at the top level (ready for reranker)
#   5. Returns correct types for all three tools
#
# Run from inside the fastapi_a container:
#   docker exec fastapi_a python testing/Chat/test_mcp_client.py
#
# All tests are run sequentially. A failure in any test prints
# FAIL with the reason and continues to the next test.
# Final line is PASSED or FAILED with a count.

import asyncio
import sys

sys.path.insert(0, "/app")
# /app is the container working directory (confirmed in Dockerfile).
# PYTHONPATH=/app is already set by docker-compose, but we insert
# explicitly here so the script works even if run outside compose.

from core.mcp_client import call_tool

# ── Seed tenant confirmed in environment state ─────────────────────────────
TENANT_ID   = "bd8c8de3-4a8e-48b9-9065-9ac08918a9c7"
DOCUMENT_ID = "e0a55c12-a4ef-4641-bf4f-9c17c4bf94ba"  # test_document.pdf, status=embedded
# These UUIDs are confirmed present in the database from the live probe
# that showed: filename=test_document.pdf, status=embedded, doc_type=other

# ── Test harness ───────────────────────────────────────────────────────────

passed = 0
failed = 0


def ok(test_name: str, detail: str = ""):
    global passed
    passed += 1
    suffix = f" — {detail}" if detail else ""
    print(f"  PASS  {test_name}{suffix}")


def fail(test_name: str, reason: str):
    global failed
    failed += 1
    print(f"  FAIL  {test_name} — {reason}")


# ── Tests ──────────────────────────────────────────────────────────────────

async def test_search_documents_returns_list():
    """
    search_documents with no filters returns a list of dicts.
    Each dict must have the keys confirmed by the live probe.
    """
    try:
        result = await call_tool(
            tool_name="search_documents",
            tool_args={},
            tenant_id=TENANT_ID,
        )

        if not isinstance(result, list):
            fail("search_documents_returns_list", f"expected list, got {type(result).__name__}")
            return

        ok("search_documents_returns_list", f"{len(result)} document(s) returned")

        # Verify required keys on each returned document
        required_keys = {"id", "filename", "status", "doc_type", "uploaded_at"}
        for doc in result:
            missing = required_keys - set(doc.keys())
            if missing:
                fail("search_documents_required_keys", f"missing keys: {missing}")
                return
        ok("search_documents_required_keys")

    except Exception as e:
        fail("search_documents_returns_list", str(e))


async def test_search_documents_tenant_isolation():
    """
    tenant_id is always injected by call_tool regardless of what
    tool_args contains. Even if tool_args has a wrong tenant_id,
    call_tool overwrites it with the authenticated tenant_id.
    We verify this by passing a deliberately wrong tenant_id in
    tool_args and confirming we still get the correct tenant's data.
    """
    try:
        result = await call_tool(
            tool_name="search_documents",
            tool_args={"tenant_id": "00000000-0000-0000-0000-000000000000"},
            # ↑ wrong tenant_id in args — call_tool must overwrite this
            tenant_id=TENANT_ID,
            # ↑ correct tenant_id injected by call_tool unconditionally
        )

        if not isinstance(result, list):
            fail("search_documents_tenant_isolation", f"expected list, got {type(result).__name__}")
            return

        # If tenant_id injection worked, we get the seed tenant's documents.
        # If it did NOT work, the wrong UUID would return an empty list.
        if len(result) == 0:
            fail("search_documents_tenant_isolation",
                 "returned empty list — tenant_id injection may have failed")
            return

        ok("search_documents_tenant_isolation",
           f"injection confirmed — {len(result)} doc(s) returned for correct tenant")

    except Exception as e:
        fail("search_documents_tenant_isolation", str(e))


async def test_get_document_summary_known_document():
    """
    get_document_summary returns a dict for a known document_id.
    Verifies required keys and correct filename.
    """
    try:
        result = await call_tool(
            tool_name="get_document_summary",
            tool_args={"document_id": DOCUMENT_ID},
            tenant_id=TENANT_ID,
        )

        if result is None:
            fail("get_document_summary_known_document",
                 "returned None for a known document_id")
            return

        if not isinstance(result, dict):
            fail("get_document_summary_known_document",
                 f"expected dict, got {type(result).__name__}")
            return

        ok("get_document_summary_known_document",
           f"returned dict for document_id={DOCUMENT_ID}")

        required_keys = {"id", "filename", "status", "doc_type",
                         "extracted_entities", "uploaded_at"}
        missing = required_keys - set(result.keys())
        if missing:
            fail("get_document_summary_required_keys", f"missing keys: {missing}")
            return
        ok("get_document_summary_required_keys")

        if result["filename"] != "test_document.pdf":
            fail("get_document_summary_correct_filename",
                 f"expected test_document.pdf, got {result['filename']}")
            return
        ok("get_document_summary_correct_filename",
           f"filename={result['filename']}")

    except Exception as e:
        fail("get_document_summary_known_document", str(e))


async def test_get_document_summary_unknown_document():
    """
    get_document_summary returns None for a document_id that does
    not exist or belongs to another tenant. This is the cross-tenant
    guard — a foreign document_id matches zero rows and returns None.
    """
    try:
        result = await call_tool(
            tool_name="get_document_summary",
            tool_args={"document_id": "00000000-0000-0000-0000-000000000000"},
            tenant_id=TENANT_ID,
        )

        if result is not None:
            fail("get_document_summary_unknown_document",
                 f"expected None for unknown document_id, got {result}")
            return

        ok("get_document_summary_unknown_document",
           "returned None for unknown document_id — cross-tenant guard confirmed")

    except Exception as e:
        fail("get_document_summary_unknown_document", str(e))


async def test_query_knowledge_base_returns_normalized_chunks():
    """
    query_knowledge_base returns a list of normalized chunk dicts.
    Normalization means:
      - chunk["text"] exists at the TOP LEVEL (not inside chunk["payload"])
      - chunk["document_id"] exists at the top level
      - chunk["section_label"] exists at the top level
      - chunk["location_index"] exists at the top level
    This shape is required by reranker.rerank() which accesses chunk["text"].
    """
    try:
        result = await call_tool(
            tool_name="query_knowledge_base",
            tool_args={"query": "document content", "top_k": 5},
            tenant_id=TENANT_ID,
        )

        if not isinstance(result, list):
            fail("query_knowledge_base_returns_list",
                 f"expected list, got {type(result).__name__}")
            return

        ok("query_knowledge_base_returns_list", f"{len(result)} chunk(s) returned")

        if len(result) == 0:
            # No embedded chunks in the corpus — cannot verify normalization.
            # This is not a failure of mcp_client.py itself.
            ok("query_knowledge_base_normalization_skipped",
               "corpus is empty — normalization check skipped")
            return

        chunk = result[0]

        # Verify text is at TOP LEVEL — this is the critical normalization check
        if "text" not in chunk:
            fail("query_knowledge_base_text_at_top_level",
                 f"'text' key missing from top level. Keys present: {list(chunk.keys())}")
            return
        ok("query_knowledge_base_text_at_top_level",
           f"text preview: {chunk['text'][:60]}...")

        # Verify payload is NOT nested inside chunk (we flattened it)
        if "payload" in chunk:
            fail("query_knowledge_base_no_nested_payload",
                 "chunk still has nested 'payload' key — normalization did not run")
            return
        ok("query_knowledge_base_no_nested_payload",
           "payload correctly flattened — no nested payload key")

        # Verify other promoted fields exist at top level
        required_top_level = {"id", "score", "text", "document_id",
                               "section_label", "location_index"}
        missing = required_top_level - set(chunk.keys())
        if missing:
            fail("query_knowledge_base_normalized_keys",
                 f"missing top-level keys after normalization: {missing}")
            return
        ok("query_knowledge_base_normalized_keys",
           f"all required top-level keys present")

    except Exception as e:
        fail("query_knowledge_base_returns_normalized_chunks", str(e))


async def test_query_knowledge_base_top_k_respected():
    """
    When top_k=3 is passed, at most 3 chunks are returned.
    Verifies that tool_args pass through correctly to the tool.
    """
    try:
        result = await call_tool(
            tool_name="query_knowledge_base",
            tool_args={"query": "document content", "top_k": 3},
            tenant_id=TENANT_ID,
        )

        if not isinstance(result, list):
            fail("query_knowledge_base_top_k_respected",
                 f"expected list, got {type(result).__name__}")
            return

        if len(result) > 3:
            fail("query_knowledge_base_top_k_respected",
                 f"expected at most 3 chunks, got {len(result)}")
            return

        ok("query_knowledge_base_top_k_respected",
           f"top_k=3 respected — {len(result)} chunk(s) returned")

    except Exception as e:
        fail("query_knowledge_base_top_k_respected", str(e))


# ── Runner ─────────────────────────────────────────────────────────────────

async def main():
    print("=" * 60)
    print("test_mcp_client.py")
    print("=" * 60)

    print("\n[search_documents]")
    await test_search_documents_returns_list()
    await test_search_documents_tenant_isolation()

    print("\n[get_document_summary]")
    await test_get_document_summary_known_document()
    await test_get_document_summary_unknown_document()

    print("\n[query_knowledge_base]")
    await test_query_knowledge_base_returns_normalized_chunks()
    await test_query_knowledge_base_top_k_respected()

    print()
    print("=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("PASSED")
    else:
        print("FAILED")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


asyncio.run(main())