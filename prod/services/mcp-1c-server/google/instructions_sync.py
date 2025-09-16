from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.config import settings
from db.database import SessionLocal
from db.models import Instruction

logger = logging.getLogger(__name__)

_SPREADSHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets.readonly"


def _ensure_credentials() -> Credentials:
    creds_path: Path = settings.GSHEETS_CREDS_PATH
    if not creds_path.exists():
        raise FileNotFoundError(f"Google credentials file not found: {creds_path}")
    return Credentials.from_service_account_file(
        str(creds_path), scopes=[_SPREADSHEETS_SCOPE]
    )


def _read_sheet_rows() -> List[Dict[str, Any]]:
    if not settings.GSHEETS_SHEET_ID:
        raise RuntimeError("GSHEETS_SHEET_ID is not configured")

    credentials = _ensure_credentials()
    service = build("sheets", "v4", credentials=credentials)
    sheet = (
        service.spreadsheets()
        .values()
        .get(
            spreadsheetId=settings.GSHEETS_SHEET_ID,
            range=settings.GSHEETS_RANGE,
        )
        .execute()
    )

    values = sheet.get("values", [])
    if not values:
        raise RuntimeError("Google Sheet returned no data")

    header = [str(h).strip() for h in values[0]]
    missing = [c for c in settings.GSHEETS_REQUIRED_COLUMNS if c not in header]
    if missing:
        raise RuntimeError(
            f"Google Sheet is missing required columns: {', '.join(missing)}"
        )

    rows: List[Dict[str, Any]] = []
    for raw_row in values[1:]:
        if not any(str(cell).strip() for cell in raw_row):
            continue
        row = {header[idx]: raw_row[idx] if idx < len(raw_row) else "" for idx in range(len(header))}
        rows.append(row)
    return rows


def _parse_json_field(column: str, value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text_value = "" if value is None else str(value).strip()
    if not text_value:
        return None
    try:
        return json.loads(text_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Column '{column}' contains invalid JSON: {exc}") from exc


def _parse_tags(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text_value = "" if value is None else str(value).strip()
    if not text_value:
        return []
    try:
        parsed = json.loads(text_value)
    except json.JSONDecodeError:
        return [tag.strip() for tag in text_value.split(",") if tag.strip()]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    if isinstance(parsed, str):
        parsed = parsed.strip()
        return [parsed] if parsed else []
    return []


def _row_to_instruction(row: Dict[str, Any], row_number: int) -> Instruction:
    missing_values: Sequence[str] = [
        column
        for column in settings.GSHEETS_REQUIRED_COLUMNS
        if not str(row.get(column, "")).strip()
    ]
    if missing_values:
        raise ValueError(
            f"Row {row_number} is missing values for: {', '.join(missing_values)}"
        )

    steps = _parse_json_field("steps", row.get("steps"))
    if steps is None or not isinstance(steps, (list, dict)):
        raise ValueError(f"Row {row_number} contains invalid steps payload")

    arg_schema = _parse_json_field("arg_schema", row.get("arg_schema"))
    field_map = _parse_json_field("field_map", row.get("field_map"))

    tags = _parse_tags(row.get("tags"))

    updated_by_raw = row.get("updated_by")
    updated_by = str(updated_by_raw).strip() if updated_by_raw else None

    data = {
        "entity": str(row["entity"]).strip(),
        "action": str(row["action"]).strip().lower(),
        "descr": str(row["descr"]).strip(),
        "steps": steps,
        "tags": tags,
    }

    if arg_schema is not None:
        data["arg_schema"] = arg_schema
    if field_map is not None:
        data["field_map"] = field_map
    if updated_by:
        data["updated_by"] = updated_by

    return Instruction(**data)


def _build_instructions(rows: Iterable[Dict[str, Any]]) -> List[Instruction]:
    instructions: List[Instruction] = []
    for index, row in enumerate(rows, start=2):
        instructions.append(_row_to_instruction(row, index))
    return instructions


def sync_instructions_from_google(session: Session | None = None) -> int:
    """Synchronise the instructions table with data from Google Sheets."""

    rows = _read_sheet_rows()
    instructions = _build_instructions(rows)

    owns_session = session is None
    db: Session
    if owns_session:
        db = SessionLocal()
    else:
        db = session

    try:
        logger.info("Replacing instructions with Google Sheet data", extra={"count": len(instructions)})
        db.execute(text("TRUNCATE TABLE instructions RESTART IDENTITY CASCADE"))
        if instructions:
            db.bulk_save_objects(instructions)
        db.commit()
        return len(instructions)
    except Exception:
        db.rollback()
        logger.exception("Failed to synchronise instructions from Google Sheets")
        raise
    finally:
        if owns_session:
            db.close()


__all__ = ["sync_instructions_from_google"]
