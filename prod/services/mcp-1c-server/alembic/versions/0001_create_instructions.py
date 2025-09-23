"""Create instructions table"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_create_instructions"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "instructions",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("entity", sa.Text(), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("descr", sa.Text(), nullable=False),
        sa.Column("steps", postgresql.JSONB(), nullable=False),
        sa.Column("arg_schema", postgresql.JSONB(), nullable=True),
        sa.Column("field_map", postgresql.JSONB(), nullable=True),
        sa.Column(
            "tags", postgresql.ARRAY(sa.Text()), server_default=sa.text("'{}'"), nullable=False
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_by", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "action in ('search','create','update','delete','post','unpost')", name="action_check"
        ),
        sa.UniqueConstraint("entity", "action", name="uix_instructions_entity_action"),
    )
    op.create_index(
        "idx_instructions_entity_action", "instructions", ["entity", "action"]
    )


def downgrade() -> None:
    op.drop_index("idx_instructions_entity_action", table_name="instructions")
    op.drop_table("instructions")

