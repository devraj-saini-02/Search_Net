"""
app/database.py
---------------
Async SQLAlchemy engine, session factory, and one-time database bootstrap.

Secret loading priority:
  1. python-dotenv reads .env from the repo root (local dev).
  2. os.getenv falls back to real environment variables   (Docker / prod).
"""

import logging
import os
from typing import AsyncGenerator

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

# Load .env once here — earliest-imported module in the app package.
# No-op when running inside a container with real env vars injected.
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. "
        "Add it to .env (dev) or inject it as an environment variable (prod)."
    )

engine = create_async_engine(
    DATABASE_URL,
    echo=False,          # Flip to True to log every SQL statement in dev
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Recycle stale connections automatically
)

# ---------------------------------------------------------------------------
# Session factory  (used by FastAPI dependencies AND the ETL worker)
# ---------------------------------------------------------------------------
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Keep objects usable after session.commit()
)


# ---------------------------------------------------------------------------
# Declarative base — every ORM model inherits from this
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# FastAPI dependency — yields a managed async DB session per request
# ---------------------------------------------------------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


# ---------------------------------------------------------------------------
# Bootstrap — called once from app lifespan on startup
# ---------------------------------------------------------------------------
async def create_tables() -> None:
    """
    Idempotently initialises the database:
      1. Enables required PostgreSQL extensions  (vector, pg_trgm).
      2. Creates all ORM-mapped tables           (CREATE TABLE IF NOT EXISTS).
      3. Creates the three performance indices   (CREATE INDEX IF NOT EXISTS).

    Safe to run on every container restart — every statement is idempotent.
    """
    async with engine.begin() as conn:

        # ── Extensions ───────────────────────────────────────────────────
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        logger.info("PostgreSQL extensions ensured: vector, pg_trgm")

        # ── Tables ───────────────────────────────────────────────────────
        # Import here to avoid a circular import at module load time.
        from app.models import Entity  # noqa: F401 — registers table with Base
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created / verified")

        # ── Indices  (raw SQL for precise operator-class control) ────────

        # 1. HNSW cosine index on 384-dim biography embeddings.
        #    m=16 / ef_construction=64 → balanced recall vs. build speed.
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_entity_embedding_hnsw
            ON entities
            USING hnsw (biography_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """))

        # 2. GIN index on the JSONB metadata column for fast key traversal.
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_entity_metadata_gin
            ON entities
            USING gin (metadata)
        """))

        # 3. Trigram GIN on name — enables sub-5 ms fuzzy autocomplete.
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_entity_name_trgm
            ON entities
            USING gin (name gin_trgm_ops)
        """))

        logger.info("Performance indices created / verified")