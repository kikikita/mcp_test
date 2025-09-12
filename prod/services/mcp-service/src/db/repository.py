"""Repository layer for working with instructions."""
from __future__ import annotations

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import models

ALLOWED_ACTIONS = {"search", "create", "update", "delete"}


class InstructionRepository:
    """High-level CRUD operations for :class:`Instruction`."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_instruction(self, entity: str, action: str) -> Optional[models.Instruction]:
        stmt = select(models.Instruction).where(
            models.Instruction.entity == entity,
            models.Instruction.action == action,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def upsert_instruction(
        self,
        entity: str,
        action: str,
        descr: str,
        steps: dict,
        arg_schema: Optional[dict] = None,
        field_map: Optional[dict] = None,
        tags: Optional[List[str]] = None,
        updated_by: Optional[str] = None,
    ) -> models.Instruction:
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"Unsupported action: {action}")

        instr = self.get_instruction(entity, action)
        if instr:
            instr.descr = descr
            instr.steps = steps
            instr.arg_schema = arg_schema
            instr.field_map = field_map
            instr.tags = tags or []
            instr.updated_by = updated_by
        else:
            instr = models.Instruction(
                entity=entity,
                action=action,
                descr=descr,
                steps=steps,
                arg_schema=arg_schema,
                field_map=field_map,
                tags=tags or [],
                updated_by=updated_by,
            )
            self.session.add(instr)
        self.session.commit()
        self.session.refresh(instr)
        return instr

    def list_entities(self) -> List[str]:
        stmt = select(models.Instruction.entity).distinct()
        return self.session.execute(stmt).scalars().all()
