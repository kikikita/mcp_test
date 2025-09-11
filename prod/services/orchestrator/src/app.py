from .components.agent import AIAgent
import asyncio
from fastapi import FastAPI, HTTPException
from typing import Dict, Any, List, Optional
import os

app = FastAPI(title="Orchestrator")
messages = {}
LLM_URL = os.getenv('LLM_URL', "http://host.docker.internal:8000/v1")
mcp_service_url = os.getenv('mcp_service_url', "http://mcp_service:9001")

@app.post("/query")
async def process_query(payload: dict):
    """
    Функция вызова LLM-агента
    Возвращает список messages
    :param payload:
    :return:
    """
    print(payload)
    messages = payload.get("messages")
    if not messages:
        raise HTTPException(status_code=400, detail="Missing messages")
    system = payload.get("system")


    async with AIAgent(mcp_service_url=mcp_service_url, messages=messages,
                       llm_url=LLM_URL) as bot:
        messages = await bot.ask(system=system)
        return messages[-1]
