"""
app/main.py
-----------
FastAPI application factory.

Responsibilities (only):
  - Create the FastAPI app instance with metadata.
  - Register the lifespan context (database bootstrap on startup).
  - Mount all routers.

Run locally:
  uvicorn app.main:app --reload

Run in Docker (example):
  CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.database import create_tables
from app.routers import admin, search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — runs once on startup, once on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Bootstrapping database …")
    await create_tables()
    logger.info("Database ready. Application starting.")
    yield
    logger.info("Application shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="High-Performance Hybrid Semantic Search Engine",
    description=(
        "Catalogue and discover real and fictional individuals via "
        "semantic vector search, exact metadata filters, and fuzzy autocomplete.\n\n"
        "**Admin credentials:** `admin_architect` / `unbreakable_secure_hash`"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(admin.router)
app.include_router(search.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"], summary="Service health check")
async def health() -> dict:
    return {"status": "ok"}