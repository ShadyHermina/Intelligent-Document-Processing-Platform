# testing/MCP/test_query_knowledge_base.py
#
# Phase 8, Step 6 verification.
#
# Exercises the query_knowledge_base MCP tool through FastMCP's in-memory
# Client. This tool calls embed_and_query, which makes a REAL OpenAI
# embedding call (text-embedding-3-small) and then searches Qdrant filtered
# by tenant_id. So this test spends a small OpenAI API call per query.
#
# Run inside the fastmcp container:
#   docker cp testing/MCP/test_query_knowledge_base.py fastmcp:/tmp/test_query_knowledge_base.py
#   docker exec fastmcp python /tmp/test_query_knowledge_base.py
#
# Embedded content = GUC Students Code of Conduct & Academic Integrity Policy
# (test_document.pdf, 153 chunks). Queries below target distinct sections so
# relevance is easy to eyeball.
#
# Mechanical assertions (do not depend on document content):
#   - default call returns top_k=20 results (collection has 153 chunks)
#   - every result's payload.tenant_id == seed tenant
#   - scores are descending, each in [0, 1]
#   - payload carries the 9 expected fields
#   - explicit top_k=3 returns exactly 3 results
#
# Relevance check (human judgment): the printed top chunks should be about
# the query's topic.

import asyncio
import json

from fastmcp import Client

import server

SEED_TENANT = "bd8c8de3-4a8e-48b9-9065-9ac08918a9c7"

EXPECTED_PAYLOAD_FIELDS = {
    "tenant_id", "document_id", "chunk_index", "text",
    "location_index", "section_label", "image_present",
    "doc_type", "file_type",
}


def _extract(result):
    """Unwrap FastMCP 2.3.4 call_tool result (list of TextContent) -> Python."""
    if isinstance(result, list):
        texts = [getattr(b, "text", None) for b in result]
        texts = [t for t in texts if t is not None]
        if not texts:
            return []
        return json.loads("".join(texts))
    for attr in ("data", "structured_content"):
        val = getattr(result, attr, None)
        if val is not None:
            if isinstance(val, dict) and "result" in val:
                return val["result"]
            return val
    return result


async def query(client, text, top_k=None):
    args = {"tenant_id": SEED_TENANT, "query": text}
    if top_k is not None:
        args["top_k"] = top_k
    result = await client.call_tool("query_knowledge_base", args)
    return _extract(result)


def check_mechanics(label, results, expect_count):
    print(f"[{label}]")
    print(f"  count = {len(results)} (expected {expect_count})")

    # tenant scoping
    tenants = {r["payload"].get("tenant_id") for r in results}
    print(f"  tenant_ids present = {tenants} "
          f"(expected just {{{SEED_TENANT}}})")

    # descending scores, all in [0,1]
    scores = [r["score"] for r in results]
    descending = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    in_range = all(0.0 <= s <= 1.0 for s in scores)
    print(f"  scores descending = {descending}; all in [0,1] = {in_range}")
    if scores:
        print(f"  top score = {scores[0]:.4f}, bottom = {scores[-1]:.4f}")

    # payload completeness on the first result
    if results:
        fields = set(results[0]["payload"].keys())
        missing = EXPECTED_PAYLOAD_FIELDS - fields
        print(f"  payload missing fields = "
              f"{missing if missing else 'none'}")
    print()


def show_top(results, n=3):
    for r in results[:n]:
        p = r["payload"]
        snippet = (p.get("text") or "").replace("\n", " ")[:160]
        print(f"    score={r['score']:.4f} "
              f"[{p.get('section_label')}] {snippet}")
    print()


async def main():
    async with Client(server.mcp) as client:
        # --- Query 1: attendance ---
        q1 = "student attendance rules for lectures and tutorials"
        r1 = await query(client, q1)
        check_mechanics(f"Q1 default top_k :: {q1}", r1, expect_count=20)
        print("  --- top 3 chunks (should be about attendance) ---")
        show_top(r1)

        # --- Query 2: cheating / plagiarism ---
        q2 = "penalties for cheating and plagiarism"
        r2 = await query(client, q2)
        check_mechanics(f"Q2 default top_k :: {q2}", r2, expect_count=20)
        print("  --- top 3 chunks (should be about academic integrity) ---")
        show_top(r2)

        # --- Query 3: smoking / alcohol, with explicit top_k=3 ---
        q3 = "rules about smoking and alcohol on campus"
        r3 = await query(client, q3, top_k=3)
        check_mechanics(f"Q3 top_k=3 :: {q3}", r3, expect_count=3)
        print("  --- all 3 chunks (should be about health/safety) ---")
        show_top(r3, n=3)


if __name__ == "__main__":
    asyncio.run(main())
