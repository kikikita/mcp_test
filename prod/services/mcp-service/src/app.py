"""
MCP-main API — FastAPI агрегатор MCP-серверов
"""
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
import os
from datetime import datetime

from fastapi import Depends
from sqlalchemy.orm import Session

from src.db.database import get_session
from src.db.repository import InstructionRepository

import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fastmcp import Client as MCPClient
from fastmcp.client.transports import SSETransport

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mcp-main")
MCP_GLARUS_SERVER = os.getenv('MCP_GLARUS_SERVER')


class ToolCallResponse(BaseModel):
    status: str
    details: str


class MCPServerConnection:
    """Подключение к конкретному MCP-серверу (HTTP или SSE)."""

    def __init__(
            self,
            name: str,
            connection_type: str,
            description: str,
            base_url: str,
    ) -> None:
        self.name = name
        self.connection_type = connection_type
        self.description = description
        self.base_url = base_url.rstrip("/")
        self.transport = SSETransport(url=f"{self.base_url}/sse")
        self._http_session: Optional[aiohttp.ClientSession] = None

    async def _ensure_http(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(base_url=self.base_url)
        return self._http_session

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Реальный сетевой запрос списка инструментов у MCP."""
        log.info(f"Fetching tools from {self.name}...")
        if self.connection_type == "SSE":
            async with MCPClient(self.transport) as client:
                return await client.list_tools()

        session = await self._ensure_http()
        async with session.get("/list_tools") as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """
        Вызывает конкретный инструмент.
        Важно: tool_name здесь - это уже полное имя server.tool,
        которое ожидает дочерний MCP-сервер.
        """
        if self.connection_type == "SSE":
            async with MCPClient(self.transport) as client:
                return await client.call_tool(tool_name, params)

        session = await self._ensure_http()
        payload = {"tool": tool_name, "param": params}
        print("payload", payload)
        async with session.post("/call_tool", json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def aclose(self) -> None:
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            log.info(f"Closed HTTP session for {self.name}")


class ToolAggregator:
    """
    Центральный сервис для управления серверами и маршрутизации вызовов инструментов.
    """

    def __init__(self, servers: Dict[str, MCPServerConnection], cache_ttl: int = 300):
        self.servers = servers
        self._cache_ttl = cache_ttl
        self._all_tools_cache: List[Dict[str, Any]] = []
        self._tool_to_server_map: Dict[str, str] = {}
        self._cache_expires_at: float = 0.0

    async def refresh_tools(self):
        """
        Обновляет кэш и карту инструментов, опрашивая все дочерние серверы.
        """
        log.info("Refreshing tool map...")
        new_tool_map = {}
        new_tool_list = []

        # Асинхронно опрашиваем все серверы
        tasks = [server.list_tools() for server in self.servers.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for server, tool_list_or_exc in zip(self.servers.values(), results):
            if isinstance(tool_list_or_exc, Exception):
                log.error(f"Failed to fetch tools from {server.name}: {tool_list_or_exc}")
                continue
            for tool in tool_list_or_exc:
                # # Проверяем на конфликт имен
                # if self.connection_type == "HTTP":
                #     tool_name = tool['function'].get("name")
                #     tool_function = tool['function']
                #     new_tool_list.append({
                #         "type": "function",
                #         "function": {
                #             "name": tool_name,
                #             "description": tool_function.get("description", ""),
                #             "parameters": tool_function.get("parameters", {}),
                #         }
                #     })
                # else:
                tool_name = tool.name
                new_tool_list.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    }
                })

                if tool_name in new_tool_map:
                    log.warning(
                        f"Tool name conflict: '{tool_name}' exists on both "
                        f"'{new_tool_map[tool_name]}' and '{server.name}'. "
                        f"The latter will be used."
                    )

                # Добавляем в карту и общий список
                new_tool_map[tool_name] = server.name
                # Сохраняем инструмент в формате OpenAI, без префикса сервера

        self._tool_to_server_map = new_tool_map
        self._all_tools_cache = new_tool_list
        self._cache_expires_at = time.time() + self._cache_ttl
        log.info(f"Tool map refreshed. Total tools: {len(self._all_tools_cache)}")

    async def get_all_tools(self) -> List[Dict[str, Any]]:
        """Возвращает общий список инструментов, при необходимости обновляя кэш."""
        if not self._all_tools_cache or time.time() >= self._cache_expires_at:
            await self.refresh_tools()
        return self._all_tools_cache

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Находит нужный сервер по имени инструмента и вызывает его."""
        # Обновляем карту, если она пуста (на случай, если startup не отработал)
        if not self._tool_to_server_map:
            await self.refresh_tools()

        server_name = self._tool_to_server_map.get(tool_name)
        if not server_name:
            return f"Tool '{tool_name}' not found."

        server = self.servers[server_name]

        log.info(f"Routing call for '{tool_name}' to server '{server_name}'")
        return await server.call_tool(tool_name, params)

    async def aclose(self):
        """Закрывает соединения всех серверов."""
        await asyncio.gather(*(s.aclose() for s in self.servers.values()))


# --- Настройка серверов и FastAPI ---
SERVERS_CONFIG = {
    "1С": MCPServerConnection(
        name="1С",
        connection_type="SSE",
        description="1C описание",
        base_url="http://host.docker.internal:4200",
    )
    # Здесь можно добавить другие серверы
}

# Создаем единый экземпляр агрегатора
aggregator = ToolAggregator(SERVERS_CONFIG)

app = FastAPI(title="MCP-main API")


# @app.on_event("startup")
# async def startup_event():
#     await aggregator.refresh_tools()


@app.on_event("shutdown")
async def shutdown_event():
    await aggregator.aclose()


class CallToolRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any] = {}


class InstructionOut(BaseModel):
    id: int
    entity: str
    action: str
    descr: str
    steps: Dict[str, Any]
    arg_schema: Optional[Dict[str, Any]] = None
    field_map: Optional[Dict[str, Any]] = None
    tags: List[str] = []
    updated_at: datetime
    updated_by: Optional[str] = None

    class Config:
        orm_mode = True


@app.get("/tools")
async def get_tools():
    """Возвращает плоский список всех доступных инструментов от всех серверов."""
    tools = await aggregator.get_all_tools()
    return {"status": "success", "tools": tools}


@app.get("/instructions/{entity}/{action}", response_model=InstructionOut)
def get_instruction(entity: str, action: str, session: Session = Depends(get_session)):
    repo = InstructionRepository(session)
    instr = repo.get(entity, action)
    if not instr:
        raise HTTPException(status_code=404, detail="Instruction not found")
    return instr


@app.post("/call_tool", response_model=ToolCallResponse)
async def call_tool(req: CallToolRequest):
    """Вызывает инструмент по его простому имени."""
    try:
        result = await aggregator.call_tool(req.tool_name, req.parameters)
        print('Ответ инструмента', result)
        return ToolCallResponse(status="Completed", details=str(result))
    except Exception as e:
        log.error(f"Error calling tool '{req.tool_name}': {e}", exc_info=True)
        return ToolCallResponse(status="Failed", details=str(e))
