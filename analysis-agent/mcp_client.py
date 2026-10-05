# mcp_client.py

import json
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class CustomerMCPClient:

    def __init__(
        self,
        server_url: str = "http://customer-mcp:8000/mcp"
    ):
        self.server_url = server_url


    async def list_tools(self) -> list[dict]:

        async with streamable_http_client(
            self.server_url
        ) as (read_stream, write_stream):

            async with ClientSession(
                read_stream,
                write_stream
            ) as session:

                await session.initialize()

                result = await session.list_tools()

                tools = []

                for tool in result.tools:

                    tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema
                    })

                return tools


    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any]
    ) -> Any:

        async with streamable_http_client(
            self.server_url
        ) as (read_stream, write_stream):

            async with ClientSession(
                read_stream,
                write_stream
            ) as session:

                await session.initialize()

                result = await session.call_tool(
                    tool_name,
                    arguments=arguments
                )

                # MCP structuredContent가 존재하면 우선 사용
                if getattr(result, "structuredContent", None):

                    return result.structuredContent

                # 일반 TextContent 처리
                outputs = []

                for item in result.content:

                    if getattr(item, "type", None) == "text":

                        text = item.text

                        # JSON 문자열이면 JSON 변환 시도
                        try:
                            outputs.append(json.loads(text))

                        except Exception:
                            outputs.append(text)

                if len(outputs) == 1:
                    return outputs[0]

                return outputs


    async def get_customer_info(
        self,
        customer_id: str
    ) -> Any:

        return await self.call_tool(
            "get_customer_info",
            {
                "customer_id": customer_id
            }
        )


    async def search_customer(
        self,
        customer_name: str
    ) -> Any:

        return await self.call_tool(
            "search_customer",
            {
                "customer_name": customer_name
            }
        )