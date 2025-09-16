from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

from pydantic import SecretStr


class Settings:
    """Application configuration loaded from environment variables."""

    def __init__(self) -> None:
        self._project_root = Path(__file__).resolve().parent.parent

        creds_path = os.getenv("GSHEETS_CREDS_PATH")
        if creds_path:
            candidate = Path(creds_path)
            if not candidate.is_absolute():
                candidate = self._project_root / candidate
        else:
            candidate = self._project_root / "creds.json"
        self.GSHEETS_CREDS_PATH: Path = candidate

        self.GSHEETS_SHEET_ID: str = os.getenv("GSHEETS_SHEET_ID", "")
        self.GSHEETS_RANGE: str = os.getenv("GSHEETS_RANGE", "instructions!A:Z")
        self.GSHEETS_REQUIRED_COLUMNS: Tuple[str, ...] = (
            "entity",
            "action",
            "descr",
            "steps",
        )

    @property
    def project_root(self) -> Path:
        return self._project_root

    def resolve_path(self, value: str | os.PathLike[str]) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self._project_root / path


settings = Settings()
