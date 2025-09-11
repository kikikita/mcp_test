"""Repository layer for working with instructions."""

from sqlalchemy.orm import Session

from . import models


class InstructionRepository:
    """Simple repository for Instruction model."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, entity: str, action: str):
        return (
            self.session.query(models.Instruction)
            .filter_by(entity=entity, action=action)
            .first()
        )

    def create(self, **data) -> models.Instruction:
        instr = models.Instruction(**data)
        self.session.add(instr)
        self.session.commit()
        self.session.refresh(instr)
        return instr

