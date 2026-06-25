# server.py — FastMCP server stub
#
# Phase 0 purpose: start the MCP server on Streamable HTTP transport,
# register one stub tool, confirm the container is reachable.
# No real tools. No business logic.

import os
from fastmcp import FastMCP

# FastMCP("name") creates the MCP server instance.
# The name is the server's identifier in the MCP protocol handshake.
# When an agent connects and asks "what server is this?", it receives
# this name. Choose something that reflects the service's role.
mcp = FastMCP("IDP-MCP-Server")

# INSTANCE_ID is not strictly necessary for FastMCP, but we include
# it for consistency with the FastAPI pattern and for future use
# when we want to log which container handled a tool call.
INSTANCE_ID = os.getenv("INSTANCE_ID", "fastmcp")


# ── Tool registration ─────────────────────────────────────────────
# @mcp.tool() registers the decorated function as a callable MCP tool.
# The function name becomes the tool name in the MCP protocol.
# The docstring becomes the tool's description — agents use this
# description to decide whether to call the tool for a given task.
# The type annotations become the tool's input/output schema.
#
# This is the decorator pattern FastMCP uses for all tool registration.
# Every real tool we add in later phases will follow this exact shape.
@mcp.tool()
def ping() -> str:
    """
    A no-op connectivity check tool.
    Returns a confirmation string to verify the MCP server is reachable
    and tool registration is working correctly.
    """
    return f"pong from {INSTANCE_ID}"


# ── Server startup ────────────────────────────────────────────────
# mcp.run() starts the internal HTTP server.
#
# transport="streamable-http"
#   Tells FastMCP to use the Streamable HTTP transport rather than
#   stdio. Streamable HTTP is the network-capable transport — it is
#   what allows other containers to call tools over HTTP.
#   stdio transport is for local process-to-process calls and would
#   not work inside Docker networking.
#
# host="0.0.0.0"
#   Same as every other container: bind to all interfaces so Docker
#   networking can route traffic to this process.
#
# port=8080
#   Matches the EXPOSE declaration in the Dockerfile.
#
# The if __name__ == "__main__" guard means this block only runs
# when the file is executed directly (python server.py).
# If server.py were ever imported as a module by another file,
# the server would not start automatically — preventing accidental
# double-starts. Good practice even when not strictly necessary.
if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8080
    )