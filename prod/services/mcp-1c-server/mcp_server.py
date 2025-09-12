"""Minimal MCP server exposing howto and todo tools."""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Union

from jsonschema import ValidationError, validate
from mcp.server.fastmcp import FastMCP

from odata_client import ODataClient

# Reuse instruction repository from mcp-service
SERVICE_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "mcp-service", "src")
)
sys.path.insert(0, SERVICE_SRC)
from db.engine import SessionLocal  # type: ignore
from db.repository import InstructionRepository, ALLOWED_ACTIONS  # type: ignore


mcp = FastMCP("1c-howto")


def _load_alias_map() -> Dict[str, str]:
    path = os.getenv(
        "MCP_ENTITY_ALIASES_FILE",
        os.path.join(os.path.dirname(__file__), "odata_entity_aliases.json"),
    )
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {alias: canon for canon, aliases in raw.items() for alias in aliases}
    except Exception:
        return {}


ENTITY_ALIASES = _load_alias_map()


def normalize_entity(name: str) -> str:
    return ENTITY_ALIASES.get(name, name)


def _fill_placeholders(obj: Any, params: Dict[str, Any]) -> Any:
    if isinstance(obj, str):
        for k, v in params.items():
            obj = obj.replace("{" + k + "}", str(v))
        return obj
    if isinstance(obj, list):
        return [_fill_placeholders(i, params) for i in obj]
    if isinstance(obj, dict):
        return {k: _fill_placeholders(v, params) for k, v in obj.items()}
    return obj


client = ODataClient(
    os.getenv("ODATA_BASE_URL", ""),
    username=os.getenv("ODATA_USERNAME"),
    password=os.getenv("ODATA_PASSWORD"),
    verify_ssl=False,
)


@mcp.tool(
    "howto",
    description="Fetch stored instruction for entity action",
    examples=[{"entity": "Catalog_Контрагенты", "action": "create"}],
)
def howto(
    entity: str,
    action: str,
    user_args: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    action = action.lower()
    if action not in ALLOWED_ACTIONS:
        return {"error": "validation_error", "details": "unsupported action"}
    entity = normalize_entity(entity)
    with SessionLocal() as session:
        repo = InstructionRepository(session)
        instr = repo.get_instruction(entity, action)
    if not instr:
        return {"error": "no_instruction"}

    user_args = user_args or {}
    if instr.arg_schema:
        try:
            validate(user_args, instr.arg_schema)
        except ValidationError as e:
            return {"error": "validation_error", "details": e.message}

    steps = _fill_placeholders(instr.steps, user_args)
    return {"recipe_id": instr.id, "descr": instr.descr, "steps": steps}


@mcp.tool(
    "todo",
    description="Execute OData steps returned by howto",
    examples=[
        {
            "steps": [
                {
                    "id": "s",
                    "entity": "Catalog_Контрагенты",
                    "func": "search",
                    "args": {"filter": "Code eq '1'"},
                }
            ]
        }
    ],
)
def todo(steps: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
    if isinstance(steps, dict):
        steps = [steps]
    results: List[Dict[str, Any]] = []
    outcome = "completed"
    for step in steps:
        func = step.get("func")
        entity = step.get("entity")
        args = step.get("args", {})
        sid = step.get("id")
        if func not in {"search", "create", "update", "delete", "post", "unpost"}:
            results.append({"id": sid, "status": "error", "details": "unsupported func"})
            outcome = "failed"
            break
        try:
            endpoint = client.entity(entity)
            data: Any
            if func == "search":
                resp = endpoint.get(filter=args.get("filter"))
                data = resp.values()
            elif func == "create":
                resp = endpoint.create(args.get("data", {}))
                data = resp.first()
            elif func == "update":
                resp = endpoint.update(id=args.get("id"), data=args.get("data", {}))
                data = resp.first()
            elif func == "delete":
                resp = endpoint.delete(id=args.get("id"), filter=args.get("filter"))
                data = {"status": resp.raw.status_code}
            elif func == "post":
                resp = endpoint.id(args.get("id")).Post()
                data = {"status": resp.raw.status_code}
            elif func == "unpost":
                resp = endpoint.id(args.get("id")).Unpost()
                data = {"status": resp.raw.status_code}
            status = "ok"
        except Exception as e:  # pragma: no cover - network errors
            status = "error"
            data = str(e)
        results.append({"id": sid, "status": status, "details": data})
        if status != "ok":
            outcome = "failed"
            break
        is_empty = not data
        on_cfg = step.get("on", {})
        if is_empty and on_cfg.get("empty") == "stop":
            outcome = "stopped"
            break
        if not is_empty and on_cfg.get("not_empty") == "stop":
            outcome = "stopped"
            break
    return {"steps": results, "summary": {"outcome": outcome}}


if __name__ == "__main__":  # pragma: no cover
    mcp.run()
