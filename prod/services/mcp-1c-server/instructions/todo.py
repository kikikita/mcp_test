from __future__ import annotations

import asyncio
from typing import Any, Dict, List

SUPPORTED_FUNCS = {"search", "create", "update", "delete", "post", "unpost"}


async def todo(step_or_steps: Any, client) -> Dict[str, Any]:
    """Execute provided steps via OData client."""
    steps: List[Dict[str, Any]] = step_or_steps if isinstance(step_or_steps, list) else [step_or_steps]
    results: List[Dict[str, Any]] = []
    for step in steps:
        func = step.get("func")
        if func not in SUPPORTED_FUNCS:
            raise ValueError(f"Unsupported func: {func}")
        args = step.get("args", {})
        method = getattr(client, func)
        if asyncio.iscoroutinefunction(method):
            data = await method(**args)
        else:
            data = await asyncio.to_thread(method, **args)
        results.append({"id": step.get("id"), "status": "ok", "details": data})
        on = step.get("on", {})
        empty = not data
        if empty and on.get("empty") in {"stop", "return"}:
            if on["empty"] == "return":
                return {"steps": results, "summary": {"outcome": "stopped"}}
            break
        if (not empty) and on.get("not_empty") in {"stop", "return"}:
            if on["not_empty"] == "return":
                return {"steps": results, "summary": {"outcome": "stopped"}}
            break
    return {"steps": results, "summary": {"outcome": "completed"}}
