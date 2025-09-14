from __future__ import annotations

from typing import Optional, List
from sqlalchemy import select

from .engine import SessionLocal
from .models import Instruction, ALLOWED_ACTIONS


def get_instruction(entity: str, action: str) -> Optional[Instruction]:
    """Return an instruction by entity and action."""
    with SessionLocal() as session:
        stmt = select(Instruction).where(Instruction.entity == entity, Instruction.action == action)
        return session.execute(stmt).scalar_one_or_none()


def upsert_instruction(
    entity: str,
    action: str,
    descr: str,
    steps: dict,
    arg_schema: Optional[dict] = None,
    field_map: Optional[dict] = None,
    tags: Optional[list[str]] = None,
    updated_by: Optional[str] = None,
) -> Instruction:
    """Insert or update an instruction."""
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Unsupported action: {action}")
    with SessionLocal.begin() as session:
        stmt = select(Instruction).where(Instruction.entity == entity, Instruction.action == action).with_for_update()
        obj = session.execute(stmt).scalar_one_or_none()
        if obj:
            obj.descr = descr
            obj.steps = steps
            obj.arg_schema = arg_schema
            obj.field_map = field_map
            obj.tags = tags
            obj.updated_by = updated_by
        else:
            obj = Instruction(
                entity=entity,
                action=action,
                descr=descr,
                steps=steps,
                arg_schema=arg_schema,
                field_map=field_map,
                tags=tags,
                updated_by=updated_by,
            )
            session.add(obj)
        session.flush()
        return obj


def list_entities() -> List[str]:
    """Return unique entities present in instructions."""
    with SessionLocal() as session:
        stmt = select(Instruction.entity).distinct()
        return session.execute(stmt).scalars().all()
