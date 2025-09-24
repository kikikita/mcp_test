# MCP 1C Platform

Платформа для интеллектуальной работы с данными 1С через Model Context Protocol (MCP) с использованием LLM, агрегации инструментов, сервиса извлечения текста из PDF и веб-интерфейса на Gradio. Решение объединяет OData-подключение к 1С, хранилище инструкций, интеграцию с Google Sheets и оркестрацию запросов к крупной языковой модели.

## Основные возможности

- Единый чат-интерфейс, комбинирующий ответы LLM и инструментальные вызовы к 1С через MCP.
- Двухфазная схема `howto` → `todo`, позволяющая модели запрашивать инструкции и исполнять шаги через OData API 1С.
- Извлечение текста из PDF с OCR (Tesseract или PaddleOCR) и передача результатов в диалог.
- Синхронизация инструкций из Google Sheets в Postgres и строгая фильтрация полей OData по allowlist.
- Расширяемая архитектура: в агрегатор MCP можно добавлять дополнительные провайдеры инструментов.
- Скрипты для запуска vLLM с моделью Salesforce/xLAM-2-32b-fc-r и поддержкой xLAM tool-calls.

## Архитектура
![Schema](schema.png)

## Состав репозитория

- `gradio_app_gpt_oss.py` — веб-интерфейс на Gradio, связывает пользователя, PDF-сервис и оркестратор.
- `prod/docker-compose.yml` — единое описание инфраструктуры (Postgres, MCP-сервисы, PDF, оркестратор).
- `prod/.env_example` — шаблон переменных окружения для продакшен-развёртывания.
- `prod/services/mcp-1c-server/` — основной MCP-сервер: OData-клиент 1С, инструменты `howto`/`todo`/`solve`, синхронизация инструкций.
- `prod/services/mcp-service/` — агрегатор инструментов, проксирующий вызовы LLM к MCP-серверам.
- `prod/services/orchestrator/` — FastAPI-сервис, который ведёт диалог с LLM и инициирует tool-calls.
- `prod/services/pdf-service/` — сервис извлечения текста из PDF с OCR.
- `vLLM/` — скрипты запуска vLLM и парсер xLAM tool-calls.
- `logs/` — каталог для логов LLM (создаётся при запуске скриптов).
- `README.md` — текущий документ.

## Предварительные требования

- Docker и Docker Compose (версии с поддержкой `docker compose`).
- Python 3.10+ (для локального запуска сервисов вне Docker).
- Доступ к OData интерфейсу 1С и учётные данные пользователя.
- Учётная запись Google Cloud с сервисным аккаунтом и доступом к Google Sheets.
- GPU с объёмом памяти, достаточным для модели Salesforce/xLAM-2-32b-fc-r (для vLLM), либо другой совместимой модели.
- Наличие Tesseract OCR и зависимостей OpenCV — автоматически устанавливаются в образ pdf-service, но потребуются локально при запуске без Docker.

## Настройка окружения

1. **Переменные окружения.**
   - Скопируйте `prod/.env_example` в `prod/.env` и заполните значения.
   - Для локального запуска UI можно использовать корневой файл `.env`.
2. **Google Sheets.**
   - Поместите файл сервисного аккаунта Google в `prod/services/mcp-1c-server/creds.json` (либо укажите иной путь в `GSHEETS_CREDS_PATH`).
   - Убедитесь, что лист содержит столбцы `entity`, `action`, `descr`, `steps` (+ дополнительные, если используются).
3. **Доступ к 1С.**
   - Проверьте, что `MCP_1C_BASE` указывает на рабочий OData endpoint, а пользователь имеет права на чтение/изменение объектов.
4. **LLM.**
   - Убедитесь, что запустите vLLM (см. раздел «LLM и инструментальные вызовы»).

### Переменные окружения

| Переменная | Назначение | Пример |
| --- | --- | --- |
| `MCP_1C_BASE` | Базовый URL OData (стандартный `/odata/standard.odata`) | `http://1c.example.local/UT11/odata/standard.odata` |
| `ONEC_USERNAME` / `ONEC_PASSWORD` | Учётные данные 1С | `svc_robot` |
| `ONEC_VERIFY_SSL` | Верификация SSL (true/false) | `false` |
| `MCP_1C_SSE_URL` | SSE endpoint MCP-сервера для UI | `http://localhost:9003/mcp/sse` |
| `MCP_CMD` | Локальная команда MCP (для клиентов MCP) | `python mcp_server.py stdio` |
| `LOG_LEVEL` / `LOG_FORMAT` | Уровень и формат логирования (`json` или `text`) | `INFO`, `json` |
| `DEBUG` | Расширенное логирование Python | `false` |

| Переменная | Назначение | Пример |
| --- | --- | --- |
| `DATABASE_URL` | Подключение к Postgres | `postgresql+psycopg2://user:pass@instructions_db:5432/mcp_howto` |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Настройки контейнера Postgres | `mcp_user` |
| `GSHEETS_SHEET_ID` | ID таблицы Google Sheets с инструкциями | `1AbCdEf...` |
| `GSHEETS_RANGE` | Диапазон листа для синхронизации | `instructions!A:Z` |
| `GSHEETS_CREDS_PATH` | Путь к JSON-файлу сервисного аккаунта | `creds.json` |

| Переменная | Назначение | Пример |
| --- | --- | --- |
| `LLM_SERVER_URL` / `LLM_URL` | Endpoint LLM (OpenAI совместимый API) | `http://host.docker.internal:8000/v1` |
| `OPENAI_API_KEY` / `LLM_API_KEY` | Ключ доступа (для vLLM может быть `empty`) | `empty` |
| `GRADIO_PORT` | Порт UI | `7860` |
| `LOGO_PATH` | Логотип в интерфейсе | `logo.jpg` |
| `PDF_SERVICE_URL` | Endpoint PDF-сервиса | `http://localhost:4300` |
| `OCR_ENGINE` / `OCR_LANG` | Настройки OCR (`tesseract`/`paddle`) | `tesseract`, `rus+eng` |
| `API_HOST` / `API_PORT` | Служебные параметры API (для UI) | `0.0.0.0`, `8080` |
| `MCP_GLARUS_SERVER` | Дополнительный MCP сервер (резерв) | `http://host.docker.internal:9002` |

## Быстрый старт (Docker Compose)

1. Перейдите в каталог `prod/` и создайте `.env` на основе примера.
2. Скопируйте Google `creds.json` в `services/mcp-1c-server/` (или измените путь в переменной окружения).
3. Запустите vLLM: `bash ../vLLM/start_vllm.sh` (по умолчанию модель Salesforce/xLAM-2-32b-fc-r, лог `logs/vllm.log`).
4. Выполните в каталоге `prod/`:
   ```bash
   docker compose up --build
   ```
5. Дождитесь появления логов:
   - `instructions_db` — готовность Postgres.
   - `mcp_1c_server` — сообщение `Instructions synchronised from Google Sheets`.
   - `pdf_service` — `Application startup complete`.
6. Откройте:
   - Gradio UI: `http://localhost:7860`
   - Orchestrator OpenAPI: `http://localhost:9080/docs`
   - MCP Aggregator: `http://localhost:9001/tools`
   - PDF health-check: `http://localhost:4300/health`

Остановить инфраструктуру: `docker compose down` (в том же каталоге). Данные инструкций сохраняются в volume `instructions_data`.

## Ручной запуск сервисов (для разработки)

1. Создайте и активируйте виртуальное окружение, установите зависимости:
   ```bash
   python -m venv .venv
   # Linux/Mac: source .venv/bin/activate
   # Windows PowerShell: .\.venv\Scripts\Activate.ps1

   pip install -r prod/services/mcp-1c-server/requirements.txt
   pip install -r prod/services/mcp-service/requirements.txt
   pip install -r prod/services/orchestrator/requirements.txt
   pip install -r prod/services/pdf-service/requirements.txt
   ```
2. Экспортируйте нужные переменные окружения из `.env`/`prod/.env`.
3. Запустите сервисы в отдельных терминалах:

- **MCP 1C Server**
  ```bash
  cd prod/services/mcp-1c-server
  alembic -c alembic.ini upgrade head
  python mcp_server.py
  ```
- **MCP Tool Aggregator**
  ```bash
  cd prod/services/mcp-service
  uvicorn src.app:app --host 0.0.0.0 --port 9001 --reload
  ```
- **Orchestrator**
  ```bash
  cd prod/services/orchestrator
  uvicorn src.app:app --host 0.0.0.0 --port 9080 --reload
  ```
- **PDF Service**
  ```bash
  cd prod/services/pdf-service
  python pdf_server.py
  ```
- **Gradio UI**
  ```bash
  python gradio_app_gpt_oss.py
  ```

При запуске вне Docker измените базовый URL MCP в `SERVERS_CONFIG` (`src/app.py`) с `host.docker.internal` на `http://localhost:4200`.

Убедитесь, что vLLM запущен и переменные окружения совпадают между сервисами.

## Компоненты

### MCP 1C Server (`prod/services/mcp-1c-server`)

- Основной `FastMCP`-сервер, публикующий SSE endpoint `/mcp/sse` и HTTP API через `mcp.streamable_http_app()`. В Docker Compose контейнер использует `network_mode: host` для прямого доступа к 1С.
- Инструменты:
  - `howto(entity, action, user_args)` — возвращает инструкцию и схему аргументов из Postgres.
  - `todo(steps|instruction)` — исполняет шаги (`search|get|create|update|delete|post|unpost`) через OData, собирает сводку и статусы шагов.
  - `solve(expression)` — безопасный калькулятор для арифметических выражений, доступный модели.
- OData-клиент (`odata_client.py`) содержит:
  - повторные попытки и пул соединений (`requests` + `HTTPAdapter`), поддержку Verbose v2/v3;
  - нормализацию GUID, alias-таблицу сущностей (`odata_entity_aliases.json`) и фильтрацию полей (`odata_field_allowlist.json`).
- При старте синхронизирует таблицу `instructions` из Google Sheets (см. `google/instructions_sync.py`). Любые существующие записи будут перезаписаны.
- Хранит инструкции в Postgres (`db/` + Alembic миграции).
- Логи: `/app/logs` внутри контейнера (`prod/services/mcp-1c-server/logs` на хосте).

### MCP Tool Aggregator (`prod/services/mcp-service`)

- FastAPI-приложение, агрегирующее инструменты нескольких MCP-серверов.
- Поддерживает SSE-транспорт через `fastmcp.client.SSETransport`, кеширует список инструментов на `cache_ttl` (по умолчанию 300 секунд).
- Эндпоинты:
  - `GET /tools` — список инструментов в формате OpenAI `type=function`.
  - `POST /call_tool` — маршрутизация вызова конкретного инструмента на родительский MCP.
- Расширяется добавлением новых записей в `SERVERS_CONFIG`. При локальном запуске обновите `base_url` на `http://localhost:4200`.

### Orchestrator (`prod/services/orchestrator`)

- FastAPI-сервис, реализующий слой LLM-агента (`AIAgent`).
- Управляет историей сообщений, добавляет системный промпт (`prompt.SYSTEM_PROMPT`), запрашивает доступные инструменты у агрегатора, интерпретирует tool-calls.
- Обрабатывает ошибки LLM API и возвращает пользователю сообщения с описанием проблем (ограничение rate-limit, ошибки JSON и т.д.).
- Эндпоинт: `POST /query` с полем `messages` (формат OpenAI Chat).

### PDF Service (`prod/services/pdf-service`)

- FastAPI с эндпоинтами:
  - `GET /` и `GET /health` для проверки состояния.
  - `POST /pdf/extract` — принимает файл, параметры OCR и возвращает либо список страниц, либо словари с метаданными (`source`, `text`).
- Использует `pdf_text_extractor.py`, который:
  - определяет цифровые PDF vs. сканы по порогу `min_chars_for_digital`;
  - поддерживает Tesseract и PaddleOCR, настраиваемые через форму (`ocr_engine`, `ocr_lang`, `dpi`).
- Временные файлы хранятся в `/app/temp`, загруженные файлы — в `/app/uploads`.

### Gradio UI (`gradio_app_gpt_oss.py`)

- Асинхронный чат с возможностью прикреплять PDF-файлы, интеграция с PDF service через `aiohttp`.
- Поддерживает ограничение истории (`CHAT_HISTORY_TURNS`), настраиваемую тему и логотип.
- Делегирует логическую обработку оркестратору, передавая системный промпт `SYSTEM_PROMPT`.

### LLM и инструментальные вызовы (`vLLM/`)

- Скрипт `start_vllm.sh` запускает vLLM с включённой поддержкой инструментов (`--enable-auto-tool-choice`, `--tool-call-parser xlam`).
- Файл `xlam_tool_call_parser.py` исправляет парсинг `tool_calls` под формат xLAM, что важно для корректной работы модели 32B.
- Логи сохраняются в `logs/vllm.log`, PID — `vllm/vllm.pid`. Порт по умолчанию: `8000`.

### База данных инструкций

- Postgres (`instructions_db`) хранит таблицу `instructions` с колонками `entity`, `action`, `descr`, `steps`, `arg_schema`, `field_map`, `tags`.
- Alembic миграции находятся в `alembic/` (`0001_create_instructions.py`).
- `InstructionRepository` предоставляет слой доступа к данным; сервис автоматически обновляет записи при запуске.

## Логи и диагностика

| Компонент | Путь/Команда |
| --- | --- |
| vLLM | `logs/vllm.log` |
| MCP 1C Server | `prod/services/mcp-1c-server/logs/*.log` и `docker compose logs mcp_1c_server` |
| PDF Service | `prod/services/pdf-service/logs/*.log` |
| Gradio | стандартный stdout (можно направить в файл через `LOGGING` настройки) |
| Postgres | `docker compose logs instructions_db` |

Уровень логирования управляется переменными `LOG_LEVEL` и `DEBUG`. Для детальной отладки OData можно временно снизить уровень фильтрации в `field_filter.py`.

## Миграции базы данных

- Применить миграции внутри контейнера:
  ```bash
  docker compose exec mcp_1c_server alembic -c alembic.ini upgrade head
  ```
- Создать новую ревизию:
  ```bash
  docker compose exec mcp_1c_server alembic -c alembic.ini revision -m "описание изменения"
  ```
- При локальном запуске убедитесь, что `DATABASE_URL` указывает на доступный Postgres.

## Проверка работоспособности

- Убедиться, что PDF сервис отвечает:
  ```bash
  curl http://localhost:4300/health
  ```
- Получить список инструментов MCP:
  ```bash
  curl http://localhost:9001/tools
  ```
- Проверить оркестратор (пример запроса):
  ```bash
  curl -X POST http://localhost:9080/query \
       -H "Content-Type: application/json" \
       -d '{"messages":[{"role":"user","content":"Покажи 5 последних заявок"}]}'
  ```
- Через UI загрузить PDF и отправить запрос в чат.

## Типичные проблемы

- **`Failed to synchronise instructions from Google Sheets`** — проверьте `GSHEETS_SHEET_ID`, диапазон и доступ сервисного аккаунта.
- **`ODataConnectionError` или таймауты** — проверьте URL 1С, сетевое подключение контейнера и значения `ONEC_USERNAME`/`ONEC_PASSWORD`.
- **LLM отвечает ошибкой 429/500** — убедитесь, что vLLM запущен и доступен, при необходимости перезапустите `start_vllm.sh`.
- **`Tool 'howto' not found`** — проверьте, что MCP 1C Server запустился и агрегатор обновил кэш (`GET /tools`).
- **PDF не распознан (пустой ответ)** — увеличьте `dpi` или переключитесь на `ocr_engine=paddle` (требует GPU/CPU ресурсов).

## Дальнейшие шаги

- Добавить другие MCP-сервисы в `SERVERS_CONFIG` (например, сторонние источники данных).
- Расширить Google Sheet дополнительными инструкциями и полями (`arg_schema`, `field_map`).
- Настроить CI/CD для автоматического деплоя контейнеров и прогонки тестов.
- Разработать автоматические тесты для OData операций и пайплайна обработки PDF.
