# inspect_tools.py
# Step 4 verification: dump each registered MCP tool's name + input schema.
#
# We import server.py (which registers the three tools via @mcp.tool()),
# then ask the FastMCP object for its tools. FastMCP 2.x exposes an async
# get_tools() returning {name: Tool}. We handle both that and the older
# list_tools() shape so this works regardless of the exact 2.3.4 surface.
#
# For each tool we print the name and the JSON input schema, so we can
# eyeball the parameter names and types the LLM layer will see.

import asyncio
import json

import server  # registers the tools as a side effect of import

mcp = server.mcp


def _schema_of(tool):
    # The Tool object stores the generated input schema. Across versions the
    # attribute has been called parameters / inputSchema / input_schema.
    # Try each so we do not depend on one exact name.
    for attr in ("parameters", "inputSchema", "input_schema"):
        schema = getattr(tool, attr, None)
        if schema is not None:
            return schema
    return "<no schema attribute found>"


async def main():
    # Preferred FastMCP 2.x API: async get_tools() -> dict[str, Tool]
    get_tools = getattr(mcp, "get_tools", None)
    tools = None

    if get_tools is not None:
        result = get_tools()
        # get_tools may be async (returns a coroutine) or sync (returns dict).
        if asyncio.iscoroutine(result):
            result = await result
        tools = result

    if tools is None:
        # Fallback: reach the tool manager's list_tools()
        manager = getattr(mcp, "_tool_manager", None)
        if manager is not None:
            listed = manager.list_tools()
            tools = {t.name: t for t in listed}

    if not tools:
        print("NO TOOLS FOUND — investigate")
        return

    print(f"TOOL COUNT: {len(tools)}\n")
    for name, tool in tools.items():
        print(f"=== {name} ===")
        schema = _schema_of(tool)
        try:
            print(json.dumps(schema, indent=2, default=str))
        except TypeError:
            print(str(schema))
        print()


if __name__ == "__main__":
    asyncio.run(main())
