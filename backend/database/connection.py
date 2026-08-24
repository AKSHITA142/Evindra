import os
import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.core.config import get_settings
from backend.database.base import Base

logger = logging.getLogger("datapilot.database")


def _build_engine():
    settings = get_settings()
    db_url = settings.database_url or os.getenv("DATABASE_URL", "sqlite:///./datapilot.db")

    engine_kwargs = {"echo": False}

    if db_url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # PostgreSQL / Supabase connection pooling & health checks
        engine_kwargs["pool_pre_ping"] = True
        engine_kwargs["pool_size"] = 10
        engine_kwargs["max_overflow"] = 20
        engine_kwargs["pool_recycle"] = 300

    logger.info(f"Connecting to database backend: {db_url.split('@')[-1] if '@' in db_url else db_url}")
    return create_engine(db_url, **engine_kwargs)


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./datapilot.db")
engine = _build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_engine():
    """Returns the SQLAlchemy engine instance."""
    return engine


def init_db(target_engine=None):
    """Initializes the database by creating all tables."""
    eng = target_engine or engine
    # Import all models to ensure they are registered on Base.metadata
    import backend.models.user
    import backend.models.dataset
    import backend.models.job
    import backend.models.experiment
    import backend.models.report

    Base.metadata.create_all(bind=eng)
    logger.info("Database schema initialized successfully.")


def get_db() -> Generator[Session, None, None]:
    """Dependency generator for FastAPI / service layer to acquire DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
