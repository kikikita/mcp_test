import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def get_engine_from_env():
    """Create SQLAlchemy engine using DB_* environment variables."""
    user = os.environ.get("DB_USER", "")
    password = os.environ.get("DB_PASSWORD", "")
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ.get("DB_NAME", "")
    if not all([user, password, name]):
        raise RuntimeError("Database credentials are not fully specified in environment")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    return create_engine(url)


engine = get_engine_from_env()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
