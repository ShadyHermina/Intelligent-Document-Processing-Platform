# fastapi/core/mcp_client.py
#
# HTTP client for calling FastMCP tools from FastAPI.
#
# Owns three responsibilities:
#   1. MCP session lifecycle — initialize handshake, initialized
#      notification, session termination
#   2. tenant_id injection — always injects from authenticated session,
#      never from LLM-supplied arguments
#   3. Result parsing — strips SSE framing, parses nested JSON,
#      normalizes query_knowledge_base chunk shape for the reranker
#
# Wire protocol (confirmed by live probe against FastMCP 2.3.4):
#
#   POST /mcp/  (no session id)
#     Body: initialize JSON-RPC request
#     Response headers: mcp-session-id
#     Response body: SSE — event: message / data: <json>
#
#   POST /mcp/  (with session id)
#     Body: notifications/initialized JSON-RPC notification (no id field)
#     Response: 200, empty or minimal body
#
#   POST /mcp/  (with session id)
#     Body: tools/call JSON-RPC request
#     Response body: SSE — event: message / data: <json>
#     Parsed path: json["result"]["content"][0]["text"] → JSON string
#                  → parse again → actual tool result
#
# Every call_tool() invocation opens a fresh session and closes it after.
# Sessions are not reused across calls. This matches FastMCP server.py's
# stateless-per-call design and avoids session expiry bugs.

import json
import logging

import httpx

from core.config import get_settings

logger = logging.getLogger(__name__)

# FastMCP container hostname is the Docker Compose service name.
# Port 8080 matches EXPOSE in fastmcp/Dockerfile and the healthcheck
# in docker-compose.yml. Both are structural facts we confirmed by
# live probe — not assumed.
_MCP_BASE_URL = "http://fastmcp:8080/mcp/"

# MCP protocol version negotiated during initialize handshake.
# Must match what FastMCP 2.3.4 accepts. Confirmed by live probe:
# server responded with protocolVersion "2024-11-05" in initialize result.
_PROTOCOL_VERSION = "2024-11-05"


def _parse_sse_body(body: str) -> dict:
    """
    Extract the JSON payload from a FastMCP SSE response body.

    FastMCP streamable-http responses are Server-Sent Events format:
        event: message\n
        data: {"jsonrpc":"2.0","id":1,"result":{...}}\n
        \n

    We need the JSON object on the data: line only.

    Parameters
    ----------
    body : str
        The raw response body string from FastMCP.

    Returns
    -------
    dict
        The parsed JSON-RPC response object.

    Raises
    ------
    ValueError
        If no data: line is found — indicates unexpected response format.
    """
    for line in body.splitlines():
        # splitlines() handles \n, \r\n, and \r correctly.
        # We iterate every line looking for the data: prefix.
        if line.startswith("data:"):
            json_str = line[len("data:"):].strip()
            # Slice off "data:" prefix, strip any leading/trailing whitespace.
            # What remains is the raw JSON string confirmed by live probe:
            # {"jsonrpc":"2.0","id":2,"result":{"content":[...],"isError":false}}
            return json.loads(json_str)

    raise ValueError(f"No data: line found in SSE response body: {body[:200]}")
    # Include first 200 chars of body in the error for debugging.
    # This should never happen with a healthy FastMCP server.


def _extract_tool_result(rpc_response: dict, tool_name: str) -> any:
    """
    Extract the actual tool result from the JSON-RPC response envelope.

    The MCP protocol wraps tool results in two layers:
      Layer 1: JSON-RPC envelope  → rpc_response["result"]
      Layer 2: MCP content array → result["content"][0]["text"]
      Layer 3: Tool return value → json.loads(text)

    The tool's Python return value (list[dict] or dict or None) was
    JSON-serialized by FastMCP when the tool returned it, and is now
    a string inside content[0]["text"]. We parse it back here.

    Parameters
    ----------
    rpc_response : dict
        The parsed JSON-RPC response from _parse_sse_body().
    tool_name : str
        Used only for error messages — identifies which tool failed.

    Returns
    -------
    any
        The tool's actual return value: list[dict] for search_documents
        and query_knowledge_base, dict or None for get_document_summary.

    Raises
    ------
    RuntimeError
        If the MCP response contains an error, or if result structure
        is not what we expect.
    """
    # Check for JSON-RPC level error first.
    if "error" in rpc_response:
        error = rpc_response["error"]
        raise RuntimeError(
            f"MCP tool '{tool_name}' returned error "
            f"code={error.get('code')} message={error.get('message')}"
        )
    # JSON-RPC errors have "error" key instead of "result" key.
    # This covers both protocol errors (wrong session, malformed request)
    # and tool execution errors that FastMCP surfaces as JSON-RPC errors.

    result = rpc_response.get("result", {})
    # result is the MCP-level result object:
    # {"content": [{"type": "text", "text": "..."}], "isError": false}

    if result.get("isError", False):
        # isError=true means the tool raised a Python exception.
        # The exception message is in content[0]["text"].
        content = result.get("content", [{}])
        error_text = content[0].get("text", "unknown error") if content else "unknown error"
        raise RuntimeError(
            f"MCP tool '{tool_name}' execution error: {error_text}"
        )

    content = result.get("content", [])
    if not content:
        # Tool returned nothing — valid for get_document_summary when
        # document is not found (returns None → serialized as "null").
        return None

    text = content[0].get("text", "")
    # content[0]["text"] is the JSON-serialized tool return value.
    # Confirmed by live probe: a JSON string containing the list of docs.

    return json.loads(text)
    # Parse the inner JSON string back to the Python object the tool returned.
    # For search_documents: list[dict]
    # For query_knowledge_base: list[dict] (each with id, score, payload)
    # For get_document_summary: dict or None


def _normalize_qkb_chunks(raw_chunks: list[dict]) -> list[dict]:
    """
    Normalize query_knowledge_base results for the reranker.

    query_knowledge_base returns chunks in this shape (from qdrant_store.py):
        {
            "id":      "<qdrant point id>",
            "score":   0.87,
            "payload": {
                "text":           "...",
                "document_id":    "...",
                "chunk_index":    3,
                "location_index": 2,
                "section_label":  "Payment Terms",
                "image_present":  false,
                "doc_type":       "contract",
                "file_type":      "pdf",
                "tenant_id":      "..."
            }
        }

    reranker.rerank() expects chunk["text"] at the TOP LEVEL of each dict,
    not nested inside chunk["payload"]["text"]. This function flattens the
    shape so the reranker can access chunk["text"] directly, while keeping
    all payload fields accessible for citation formatting in chat.py.

    The flattened shape is:
        {
            "id":             "<qdrant point id>",
            "score":          0.87,
            "text":           "...",          ← promoted to top level
            "document_id":    "...",          ← promoted to top level
            "chunk_index":    3,              ← promoted to top level
            "location_index": 2,              ← promoted to top level
            "section_label":  "Payment Terms",← promoted to top level
            "image_present":  false,          ← promoted to top level
            "doc_type":       "contract",     ← promoted to top level
            "file_type":      "pdf",          ← promoted to top level
        }
        (tenant_id is dropped — not needed downstream)

    Parameters
    ----------
    raw_chunks : list[dict]
        Raw output from query_knowledge_base tool call.

    Returns
    -------
    list[dict]
        Flattened chunk dicts ready for reranker.rerank() and citation
        formatting in chat.py.
    """
    normalized = []
    for chunk in raw_chunks:
        payload = chunk.get("payload", {})
        normalized.append({
            "id":             chunk.get("id"),
            "score":          chunk.get("score"),
            # Promote all payload fields to top level:
            "text":           payload.get("text", ""),
            "document_id":    payload.get("document_id"),
            "chunk_index":    payload.get("chunk_index"),
            "location_index": payload.get("location_index"),
            "section_label":  payload.get("section_label"),
            "image_present":  payload.get("image_present"),
            "doc_type":       payload.get("doc_type"),
            "file_type":      payload.get("file_type"),
            # tenant_id intentionally excluded — not needed after this point.
        })
    return normalized


async def call_tool(
    tool_name: str,
    tool_args: dict,
    tenant_id: str,
) -> any:
    """
    Call a FastMCP tool over HTTP and return the parsed result.

    Handles the full MCP session lifecycle per call:
      1. Initialize session  → get mcp-session-id
      2. Send initialized notification
      3. Call the tool
      4. Parse and return the result
      5. Terminate the session (best-effort cleanup)

    tenant_id is always injected here from the authenticated session.
    Whatever tool_args contains (from the LLM's tool_call), tenant_id
    is overwritten unconditionally. The LLM cannot supply or influence
    the tenant_id value.

    For query_knowledge_base results, chunks are normalized to flat
    dicts with text at the top level, ready for reranker.rerank().

    Parameters
    ----------
    tool_name : str
        One of: search_documents, query_knowledge_base, get_document_summary.
    tool_args : dict
        Arguments from the LLM's tool_call, with tenant_id absent
        (excluded from the OpenAI tool schema) or ignored if present.
    tenant_id : str
        The authenticated tenant UUID from the WebSocket session.
        Always injected unconditionally.

    Returns
    -------
    any
        Parsed tool result. Type depends on the tool:
        - search_documents:      list[dict]
        - query_knowledge_base:  list[dict] (normalized, flat)
        - get_document_summary:  dict | None

    Raises
    ------
    RuntimeError
        If session initialization fails, tool call fails, or the
        server returns an unexpected response format.
    httpx.HTTPError
        If the HTTP connection to fastmcp fails entirely.
    """

    # --- Inject tenant_id unconditionally ---
    tool_args["tenant_id"] = tenant_id
    # This is the enforcement point described in the Phase 9 design.
    # Overwrites any tenant_id the LLM may have supplied (it cannot,
    # because tenant_id is excluded from the OpenAI tool schema, but
    # we overwrite defensively regardless).

    async with httpx.AsyncClient(timeout=30.0) as client:
        # AsyncClient is used because call_tool is async — we await
        # every HTTP call. A single client instance is used for all
        # three requests (init, notification, tool call) so they share
        # the same underlying TCP connection where possible.
        # timeout=30.0 seconds covers the OpenAI embedding call inside
        # query_knowledge_base, which is the slowest operation.
        # Alternative: separate timeouts per step. Overkill at this phase.

        # ── Step 1: Initialize session ────────────────────────────────
        init_response = await client.post(
            _MCP_BASE_URL,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {
                        "name": "idp-fastapi",
                        "version": "1.0.0",
                    },
                },
            },
            headers={
                "Accept": "application/json, text/event-stream",
            },
        )
        init_response.raise_for_status()
        # raise_for_status() raises httpx.HTTPStatusError on 4xx/5xx.
        # If FastMCP is unreachable or returns an error here, the
        # exception propagates to the caller (chat.py orchestration loop)
        # which handles it as a tool call failure.

        session_id = init_response.headers.get("mcp-session-id")
        if not session_id:
            raise RuntimeError(
                "FastMCP initialize response missing mcp-session-id header"
            )
        # Confirmed by live probe: mcp-session-id is always present on
        # a successful initialize response. Its absence means the
        # initialization failed silently — treat as an error.

        logger.debug(f"MCP session initialized: {session_id}")

        # ── Step 2: Send initialized notification ─────────────────────
        notif_response = await client.post(
            _MCP_BASE_URL,
            json={
                "jsonrpc": "2.0",
                # No "id" field — this is a JSON-RPC notification,
                # not a request. Notifications do not expect a response.
                # Including an "id" would make it a request and the
                # server would wait to send a response, changing semantics.
                "method": "notifications/initialized",
                "params": {},
            },
            headers={
                "Accept": "application/json, text/event-stream",
                "mcp-session-id": session_id,
            },
        )
        notif_response.raise_for_status()
        # The server returns 200 with an empty or minimal body.
        # We do not parse the body — notifications have no result.

        # ── Step 3: Call the tool ──────────────────────────────────────
        tool_response = await client.post(
            _MCP_BASE_URL,
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": tool_args,
                    # tool_args now contains tenant_id (injected above)
                    # plus whatever the LLM supplied for the other params.
                },
            },
            headers={
                "Accept": "application/json, text/event-stream",
                "mcp-session-id": session_id,
            },
        )
        tool_response.raise_for_status()

        # ── Step 4: Parse result ───────────────────────────────────────
        rpc_response = _parse_sse_body(tool_response.text)
        result = _extract_tool_result(rpc_response, tool_name)

        logger.info(
            f"MCP tool '{tool_name}' called successfully. "
            f"Result type: {type(result).__name__}, "
            f"items: {len(result) if isinstance(result, list) else 'n/a'}"
        )

        # ── Step 5: Normalize query_knowledge_base chunks ─────────────
        if tool_name == "query_knowledge_base" and isinstance(result, list):
            result = _normalize_qkb_chunks(result)
            # Flatten payload fields to top level so reranker.rerank()
            # can access chunk["text"] directly.
            # Only applied to query_knowledge_base — the other two tools
            # return flat dicts already.

        # ── Step 6: Terminate session (best-effort) ───────────────────
        try:
            await client.delete(
                _MCP_BASE_URL,
                headers={"mcp-session-id": session_id},
            )
        except Exception:
            # Session termination is best-effort. If the DELETE fails
            # (network hiccup, server restarted), we log and continue.
            # The session will expire server-side eventually.
            # We never let cleanup failure propagate to the caller.
            logger.debug(f"MCP session cleanup failed for {session_id} — ignored")

        return result