"""An MCP client that starts the server and calls its tool (chapter 39).

Usage: uv run --group rag --group neural python -m guide_nlp.chapter39.mcp_client
"""

import asyncio
import sys

from mcp import Client
from mcp.client.stdio import StdioServerParameters


async def main() -> None:
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "guide_nlp.chapter39.mcp_server"]
    )
    async with Client(server) as client:
        print((await client.list_tools()).tools)
        result = await client.call_tool("chercher", {"requete": "planétarium", "k": 2})
        print(result.structured_content)


if __name__ == "__main__":
    asyncio.run(main())
