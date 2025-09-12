"""Database engine and session management for instructions service."""

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


def get_engine_from_env():
    """Create SQLAlchemy engine from DB_* environment variables."""
    user = os.getenv("DB_USER", "mcp_user")
    password = os.getenv("DB_PASS", "mcp_pass")
    host = os.getenv("DB_HOST", "instructions_db")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "mcp_howto")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    return create_engine(url, future=True)


engine = get_engine_from_env()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


class Base(DeclarativeBase):
    """Base declarative class for ORM models."""


def get_session() -> Generator[Session, None, None]:
    """Yield a database session and ensure closure."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

