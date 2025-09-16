import asyncio
import logging
from contextlib import asynccontextmanager
from google import genai
import threading
from core.config import settings

logger = logging.getLogger(__name__)


class ApiKeyPool:
    """Manage Google API keys with round-robin selection."""

    def __init__(self) -> None:
        self._keys: list[str] | None = None
        self._index = 0
        self._lock = asyncio.Lock()
        self._sync_lock = threading.Lock()

    def _load_keys(self) -> None:
        keys_raw = settings.gemini_api_key
        keys_str = keys_raw.get_secret_value()
        keys = [k.strip() for k in keys_str.split(',') if k.strip()] if keys_str else []
        if not keys:
            msg = "Google API keys are not configured or invalid"
            logger.error(msg)
            raise ValueError(msg)
        self._keys = keys

    async def get_key(self) -> str:
        async with self._lock:
            if self._keys is None:
                self._load_keys()
            key = self._keys[self._index]
            self._index = (self._index + 1) % len(self._keys)
            logger.debug("Using Google API key index %s", self._index)
            return key

    def get_key_sync(self) -> str:
        """Synchronous helper for environments without an event loop."""
        with self._sync_lock:
            if self._keys is None:
                self._load_keys()
            key = self._keys[self._index]
            self._index = (self._index + 1) % len(self._keys)
            logger.debug("Using Google API key index %s", self._index)
            return key


class GoogleClientFactory:
    """Factory for thread-safe creation of Google GenAI clients."""

    _pool = ApiKeyPool()

    @classmethod
    @asynccontextmanager
    async def image(cls):
        key = await cls._pool.get_key()
        client = genai.Client(api_key=key)
        try:
            yield client.aio
        finally:
            pass

    @classmethod
    @asynccontextmanager
    async def audio(cls):
        key = await cls._pool.get_key()
        client = genai.Client(api_key=key, http_options={"api_version": "v1alpha"})
        try:
            yield client.aio
        finally:
            pass
