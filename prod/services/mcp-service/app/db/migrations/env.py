"""Alembic environment configuration."""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

# Ensure the service's ``src`` package is importable when Alembic loads this module.
# ``env.py`` lives in ``app/db/migrations``; go up three directories to the service
# root and then append ``src`` to ``sys.path``.
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

from db.database import Base  # noqa: E402
from db import models  # noqa: E402

from alembic import context
from sqlalchemy import engine_from_config, pool

config = context.config

config.set_main_option(
    "sqlalchemy.url",
    os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://mcp_user:mcp_pass@instructions_db:5432/mcp_howto",
    ),
)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
