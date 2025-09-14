from __future__ import annotations

import re
from typing import Any, Dict

from jsonschema import validate, ValidationError as JsonValidationError

from .repository import get_instruction


class InstructionNotFoundError(Exception):
    """Raised when instruction not found."""


class UserArgsValidationError(Exception):
    """Raised when provided user_args do not match schema."""


def _replace_placeholders(value: Any, user_args: Dict[str, Any]) -> Any:
    """Recursively replace {placeholders} in value using user_args."""
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            return str(user_args.get(key, match.group(0)))

        return re.sub(r"{([^{}]+)}", repl, value)
    if isinstance(value, list):
        return [_replace_placeholders(v, user_args) for v in value]
    if isinstance(value, dict):
        return {k: _replace_placeholders(v, user_args) for k, v in value.items()}
    return value


def howto(entity: str, action: str, user_args: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return recipe for given entity/action."""
    inst = get_instruction(entity, action)
    if not inst:
        raise InstructionNotFoundError("no_instruction")
    user_args = user_args or {}
    if inst.arg_schema:
        try:
            validate(instance=user_args, schema=inst.arg_schema)
        except JsonValidationError as exc:  # pragma: no cover - thin wrapper
            raise UserArgsValidationError(str(exc)) from exc
    steps = _replace_placeholders(inst.steps, user_args)
    return {"recipe_id": inst.id, "descr": inst.descr, "steps": steps}
