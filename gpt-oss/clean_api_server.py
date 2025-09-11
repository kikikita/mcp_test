#!/usr/bin/env python3
import os
import json
import logging
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.mcp_server.prompt import SYSTEM_PROMPT

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
API_PORT = int(os.getenv("API_PORT", "8080"))
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:9080/query")
MAX_TURNS = 20

# Хранилище сессий
chat_sessions: Dict[str, Dict] = {}

# Модели Pydantic
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

class SessionInfo(BaseModel):
    id: str
    title: str
    created_at: str
    message_count: int

# Создание приложения
app = FastAPI(title="AI Финансист API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Функции работы с сессиями
def create_session() -> str:
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = {
        'id': session_id,
        'messages': [],
        'created_at': datetime.now().isoformat(),
        'title': 'Новый чат'
    }
    logger.info(f"Created session: {session_id}")
    return session_id

def add_message(session_id: str, role: str, content: str):
    if session_id not in chat_sessions:
        session_id = create_session()
    
    chat_sessions[session_id]['messages'].append({
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat()
    })
    
    # Обновляем заголовок на основе первого сообщения
    user_messages = [m for m in chat_sessions[session_id]['messages'] if m['role'] == 'user']
    if len(user_messages) == 1 and role == 'user':
        title = content[:50] + "..." if len(content) > 50 else content
        chat_sessions[session_id]['title'] = title
    
    # Ограничиваем историю
    if len(chat_sessions[session_id]['messages']) > MAX_TURNS * 2:
        chat_sessions[session_id]['messages'] = chat_sessions[session_id]['messages'][-MAX_TURNS * 2:]

async def call_orchestrator(messages: List[Dict]) -> str:
    try:
        request_data = {
            'messages': messages,
            'system': SYSTEM_PROMPT
        }
        
        logger.info(f"Calling orchestrator: {json.dumps(request_data, ensure_ascii=False)}")
        
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(ORCHESTRATOR_URL, json=request_data) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Orchestrator response: {data}")
                    return data.get("content", "Пустой ответ от модели")
                else:
                    error = await response.text()
                    logger.error(f"Orchestrator error {response.status}: {error}")
                    return "Ошибка при обращении к модели"
    except Exception as e:
        logger.error(f"Orchestrator call failed: {e}")
        return f"Ошибка подключения: {str(e)}"

# API Endpoints
@app.get("/")
async def root():
    return {
        "name": "AI Финансист API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/health", "/api/chat", "/api/sessions"]
    }

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Сервер работает"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Chat request: {request.message}")
        
        # Создаем сессию если нет
        session_id = request.session_id
        if not session_id or session_id not in chat_sessions:
            session_id = create_session()
        
        # Добавляем сообщение пользователя
        add_message(session_id, "user", request.message)
        
        # Подготавливаем историю для модели
        session_messages = chat_sessions[session_id]['messages']
        messages_for_model = []
        
        for msg in session_messages[:-1]:  # Все кроме последнего
            messages_for_model.append({
                'role': msg['role'],
                'content': msg['content']
            })
        
        # Добавляем текущее сообщение
        messages_for_model.append({
            'role': 'user',
            'content': request.message
        })
        
        # Вызываем модель
        response_text = await call_orchestrator(messages_for_model)
        
        # Сохраняем ответ
        add_message(session_id, "assistant", response_text)
        
        return ChatResponse(response=response_text, session_id=session_id)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions", response_model=List[SessionInfo])
async def get_sessions():
    sessions = []
    for session_id, session_data in chat_sessions.items():
        sessions.append(SessionInfo(
            id=session_id,
            title=session_data['title'],
            created_at=session_data['created_at'],
            message_count=len(session_data['messages'])
        ))
    return sorted(sessions, key=lambda x: x.created_at, reverse=True)

@app.post("/api/sessions")
async def create_new_session():
    session_id = create_session()
    return {"session_id": session_id, "message": "Сессия создана"}

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return chat_sessions[session_id]

if __name__ == "__main__":
    logger.info(f"Starting API server on port {API_PORT}")
    logger.info(f"Orchestrator URL: {ORCHESTRATOR_URL}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=API_PORT,
        log_level="info"
    )