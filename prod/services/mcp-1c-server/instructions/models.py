from __future__ import annotations

from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, func, UniqueConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()

ALLOWED_ACTIONS = {"search", "create", "update", "delete"}


class Instruction(Base):
    __tablename__ = "instructions"
    __table_args__ = (UniqueConstraint("entity", "action", name="uix_entity_action"),)

    id = Column(Integer, primary_key=True)
    entity = Column(String, nullable=False)
    action = Column(String, nullable=False)
    descr = Column(Text, nullable=False)
    steps = Column(JSON, nullable=False)
    arg_schema = Column(JSON)
    field_map = Column(JSON)
    tags = Column(JSON)
    updated_by = Column(String)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:  # pragma: no cover - simple debug helper
        return f"<Instruction entity={self.entity!r} action={self.action!r}>"
