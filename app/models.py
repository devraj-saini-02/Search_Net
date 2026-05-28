"""
app/models.py
-------------
SQLAlchemy 2.0 ORM model for the unified `entities` table.

Design notes:
  - Python attribute `entity_metadata` maps to DB column `metadata`.
    The rename avoids shadowing SQLAlchemy's own `DeclarativeBase.metadata`.
  - Performance indices are created via raw SQL in database.create_tables(),
    not here, to benefit from `CREATE INDEX IF NOT EXISTS` idempotency and
    precise operator-class syntax (vector_cosine_ops, gin_trgm_ops).
  - The composite UNIQUE(name, is_fictional) is the ETL upsert conflict key:
    "Batman (fictional)" and "Batman (real person)" are distinct records.
"""

import uuid

from sqlalchemy import Boolean, Column, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from pgvector.sqlalchemy import Vector

from app.database import Base


class Entity(Base):
    __tablename__ = "entities"

    # ── Primary key ──────────────────────────────────────────────────────
    id = Column(
        PGUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

    # ── Core identity ────────────────────────────────────────────────────
    name = Column(String(255), nullable=False)
    is_fictional = Column(Boolean, nullable=False)

    # ── Biography (clean intro paragraph from Wikipedia) ─────────────────
    biography = Column(Text, nullable=True)

    # ── 384-dim semantic vector (matches all-MiniLM-L6-v2 output) ────────
    biography_embedding = Column(Vector(384), nullable=True)

    # ── Polymorphic metadata tree stored as JSONB ─────────────────────────
    # Real person:   {"profession": "Physicist", "timeline": {"born": 1879}}
    # Fictional:     {"universe": "Marvel", "creators": ["Stan Lee"]}
    # Python attr:   entity_metadata
    # DB column:     metadata
    entity_metadata = Column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    # ── Poster / profile image URL (from TMDB for fictional entities) ─────
    image_url = Column(String, nullable=True)

    # ── Constraints ───────────────────────────────────────────────────────
    __table_args__ = (
        UniqueConstraint(
            "name",
            "is_fictional",
            name="uq_entity_name_fictional",
        ),
    )

    def __repr__(self) -> str:
        flag = "fictional" if self.is_fictional else "real"
        return f"<Entity id={self.id} name={self.name!r} ({flag})>"