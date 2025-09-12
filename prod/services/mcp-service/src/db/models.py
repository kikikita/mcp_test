"""ORM models for the MCP service."""

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    Index,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP
from sqlalchemy.sql import text

from .engine import Base


class Instruction(Base):
    """Represents an instruction for orchestrating actions."""

    __tablename__ = "instructions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    entity = Column(Text, nullable=False)
    action = Column(Text, nullable=False)
    descr = Column(Text, nullable=False)
    steps = Column(JSONB, nullable=False)
    arg_schema = Column(JSONB, nullable=True)
    field_map = Column(JSONB, nullable=True)
    tags = Column(ARRAY(Text), server_default=text("'{}'"))
    updated_at = Column(
        TIMESTAMP(timezone=True), server_default=text("now()"), nullable=False
    )
    updated_by = Column(Text, nullable=True)

    __table_args__ = (
        CheckConstraint(
            "action in ('search','create','update','delete')", name="action_check"
        ),
        Index("idx_instructions_entity_action", "entity", "action"),
        Index(
            "uix_instructions_entity_action",
            "entity",
            "action",
            unique=True,
        ),
    )

