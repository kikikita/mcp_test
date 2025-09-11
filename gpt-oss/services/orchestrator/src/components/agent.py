import asyncio
import json
import os
from fastmcp import Client as MCP
from fastmcp.client.transports import SSETransport
from openai import AsyncOpenAI, APIStatusError, APIConnectionError
from typing import Dict, Any, List, Optional, Literal
import httpx
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
import requests


class FunctionDefinition(BaseModel):
    """
    Описывает структуру 'function' внутри инструмента.
    """
    name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    """
    Описывает полную структуру инструмента в формате OpenAI.
    """
    type: Literal["function"]
    function: FunctionDefinition


tools_validator = TypeAdapter(List[ToolDefinition])


class AIAgent:
    """
    Оркестратор, соединяющий LLM с MCP-сервером (FastAPI).
    MCP-сервер предоставляет два эндпоинта:
      • GET  /tools      → список инструментов
      • POST /call_tool  → вызов инструмента
    """

    def __init__(
            self,
            mcp_service_url: str,  # напр. "http://localhost:9001"
            session_id: str='',
            llm_url: str = "http://host.docker.internal:8000/v1",
            model: str = "",
            messages=None
    ) -> None:
        if messages is None:
            messages = []
        self._mcp_url = mcp_service_url.rstrip("/")
        self.session_id = session_id
        self.url = llm_url
        api_key = os.getenv("LLM_API_KEY", "empty")
        self.llm = AsyncOpenAI(base_url=self.url, api_key=api_key)
        self.model = model
        self.messages = messages
        self.tools: List[Dict[str, Any]] = []  # список в формате OpenAI

    # ----------------------- контекст-менеджер -----------------------
    async def __aenter__(self):
        await self.llm.__aenter__()
        self.tools = await self._list_tools()
        return self

    async def __aexit__(self, *exc):
        await self.llm.__aexit__(*exc)

    # ----------------------- MCP-служебные методы --------------------
    #def _push_reasoning(self, chunk: str):
        #"""
        #Отправка reasoning по WS
        #"""
        #requests.post(
        #    "http://ws:9090/reasoning",
        #    json={"session_id": self.session_id, "data": chunk}
        #)

    async def _list_tools(self):
        """
        Запрашивает список инструментов у MCP и приводит
        в формат, ожидаемый OpenAI (`type=function`, ...).
        """
        resp = requests.get(self._mcp_url+"/tools")
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            raise RuntimeError("MCP /tools ожидает status=success")
        return data.get('tools')

    async def _call_tool(self, name: str, params: Dict[str, Any]) -> Any:
        """
        Делает POST /call_tool и возвращает JSON-ответ MCP.
        """
        #self._push_reasoning(f'Вызван инструмент {name}')
        payload = {"tool_name": name, "parameters": params}
        resp = requests.post(self._mcp_url+"/call_tool", json=payload)
        resp.raise_for_status()
        return resp.json()

    # ----------------------- основной публичный метод ----------------
    async def ask(self, system: str | None = None) -> str:
        """
        Отправляет единственный запрос к LLM, автоматически
        обрабатывая все tool-calls.
        """
        #TODO: Добавить историю диалогов
        if not {"role": "system", "content": system} in self.messages:
            self.messages = [{"role": "system", "content": system}] + self.messages

        while True:
            try:
                print("Отправляю запрос...")
                resp = await self.llm.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tools,
                    tool_choice="auto"
                )
            except APIStatusError as e:
                # Ошибка от API, например 500, 429, 400 и т.д.
                # Библиотека openai оборачивает httpx.HTTPStatusError в свой класс
                print(f"Ошибка от API LLM: статус {e.status_code}, ответ: {e.response.text}")
                if e.status_code == 500:
                    error_msg = "Ошибка: LLM-сервер вернул внутреннюю ошибку (500)."
                else:
                    # Здесь можно добавить обработку других кодов, например, 429 (rate limit)
                    # или просто прервать выполнение.
                    error_msg = "Ошибка: Не удалось получить ответ от LLM."
                self.messages.append({"role": "assistant", "content": error_msg})
                return self.messages

            except APIConnectionError as e:
                # Ошибка соединения (не удалось подключиться к серверу LLM)
                # Оборачивает httpx.ConnectError
                print(f"Ошибка подключения к LLM: {e.__cause__}")
                error_msg = "Ошибка: Не удалось подключиться к LLM-серверу."
                self.messages.append({"role": "assistant", "content": error_msg})
                return self.messages

            except Exception as e:
                # Этот блок теперь будет ловить другие, более редкие ошибки.
                # Например, если LLM вернула ответ, но он не соответствует ожиданиям
                # (например, невалидный JSON в tool_calls.function.arguments).

                print(f"Произошла неожиданная ошибка при обработке ответа LLM: {e}")

                # Ваш код для обработки "плохого" ответа ассистента
                if not self.messages or self.messages[-1].get("role") != "assistant":
                    # Если последний ответ не от ассистента, логика может не сработать.
                    # Добавим защиту.
                    print("Не удалось применить логику исправления, последнее сообщение не от ассистента.")
                    error_msg = "Ошибка: Не удалось обработать ответ LLM."
                    self.messages.append({"role": "assistant", "content": error_msg})
                    return self.messages

                last = self.messages.pop()
                tool_call_id = "fault"
                if last.get("tool_calls"):
                    tool_call_id = last["tool_calls"][0]["id"]

                content_to_fix = str(last.get("content", ""))
                cut_content = content_to_fix[:len(content_to_fix) // 2]

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"Ошибка при обработке предыдущего ответа. Сокращенное содержимое: {cut_content}",
                    }
                )
                continue

            self.messages.append(resp.choices[0].message.model_dump())
            msg = resp.choices[0].message
            print("msg", msg)
            if tool_calls := self.messages[-1].get("tool_calls", None):
                for tool_call in tool_calls:
                    call_id: str = tool_call["id"]
                    if fn_call := tool_call.get("function"):
                        fn_name: str = fn_call["name"]
                        fn_args: dict = json.loads(fn_call["arguments"])
                        result = await self._call_tool(fn_name, fn_args)

                        self.messages.append({
                            "role": "tool",
                            "content": str(result),
                            "tool_call_id": call_id,
                        })
                continue
            return self.messages
