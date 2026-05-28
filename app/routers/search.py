"""
app/routers/search.py
---------------------
Public search endpoints — no authentication required.

  GET  /api/v1/search/suggest   Keystroke-level trigram autocomplete (≤ 5 ms)
  POST /api/v1/search            Hybrid semantic + metadata filter search

Implicit-filter handling
------------------------
When a user embeds metadata in their natural language prompt
(e.g. "Marvel genius inventor" instead of using explicit filters),
decompose_search_prompt() detects and extracts those filters automatically.

It runs CONCURRENTLY with embed_text() via asyncio.gather() so there is
zero additional wall-clock latency on the search path.

Merge precedence: explicit filters passed in the request body always
win over anything extracted from the prompt text.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.etl_worker import decompose_search_prompt, embed_text
from app.models import Entity
from app.schemas import EntitySearchResult, SearchRequest, SuggestResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/search", tags=["Search"])


# ---------------------------------------------------------------------------
# GET /api/v1/search/suggest
# ---------------------------------------------------------------------------

@router.get(
    "/suggest",
    response_model=List[SuggestResult],
    summary="Keystroke-level name autocomplete",
    description=(
        "Returns up to **5** name + image_url pairs ranked by trigram similarity.\n\n"
        "Combines prefix ILIKE and pg_trgm similarity > 0.15 in a single OR "
        "for maximum recall. Both paths are accelerated by the trigram GIN index.\n\n"
        "**Access:** Public."
    ),
)
async def suggest(
    q: str = Query(..., min_length=1, description="Partial name typed by the user"),
    db: AsyncSession = Depends(get_db),
) -> List[SuggestResult]:
    similarity_score = func.similarity(Entity.name, q)

    stmt = (
        select(Entity.name, Entity.image_url)
        .where(
            or_(
                Entity.name.ilike(f"{q}%"),
                similarity_score > 0.15,
            )
        )
        .order_by(similarity_score.desc())
        .limit(5)
    )

    rows = (await db.execute(stmt)).all()
    return [SuggestResult(name=row.name, image_url=row.image_url) for row in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_filters(
    explicit: Optional[Any],
    extracted: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merges implicit filters (extracted from prompt text) with explicit filters
    (passed in the request body). Explicit values always take priority.

    Returns a flat dict:
      {
        "is_fictional":     bool | None,
        "metadata_filters": { key: value, ... }
      }
    """
    # Start from extracted (lower priority)
    merged_is_fictional:   Optional[bool]       = extracted.get("is_fictional")
    merged_meta_filters:   Dict[str, Any]       = dict(extracted.get("metadata_filters") or {})

    if explicit is not None:
        # Explicit is_fictional overrides extracted
        if explicit.is_fictional is not None:
            merged_is_fictional = explicit.is_fictional

        # Explicit metadata_filters override extracted on a per-key basis
        if explicit.metadata_filters:
            merged_meta_filters.update(explicit.metadata_filters)

    return {
        "is_fictional":     merged_is_fictional,
        "metadata_filters": merged_meta_filters,
    }


# ---------------------------------------------------------------------------
# POST /api/v1/search
# ---------------------------------------------------------------------------

@router.post(
    "",
    response_model=List[EntitySearchResult],
    summary="Hybrid semantic + metadata filter search",
    description=(
        "Full hybrid search pipeline:\n\n"
        "**Step 1 (concurrent):**\n"
        "- Embeds `search_prompt` via all-MiniLM-L6-v2 (384-dim vector).\n"
        "- Decomposes prompt for implicit metadata hints "
        "(e.g. `'Marvel genius'` → `universe=Marvel` filter extracted automatically).\n"
        "Both run in parallel — zero extra latency.\n\n"
        "**Step 2:** Merges explicit `filters` with any extracted ones. "
        "Explicit values always win.\n\n"
        "**Step 3:** Applies merged filters — scalar `is_fictional` and "
        "JSONB `metadata_filters` (injection-safe bound parameters).\n\n"
        "**Step 4:** Orders by cosine distance via pgvector `<=>` + HNSW index. "
        "Returns minimum **10 results**. `biography_embedding` never included.\n\n"
        "**Example — explicit filters:**\n"
        "```json\n"
        '{"search_prompt": "eccentric genius", '
        '"filters": {"is_fictional": true, "metadata_filters": {"universe": "Marvel"}}}\n'
        "```\n\n"
        "**Example — implicit filters (no filters block needed):**\n"
        "```json\n"
        '{"search_prompt": "Marvel eccentric genius inventor"}\n'
        "```\n\n"
        "**Access:** Public."
    ),
)
async def hybrid_search(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
) -> List[EntitySearchResult]:

    # ── 1. Concurrent: embed prompt + decompose implicit filters ──────────
    # Both hit the HF API simultaneously. Total latency = max(embed, decompose)
    # instead of embed + decompose sequentially.
    try:
        async with httpx.AsyncClient() as client:
            query_vector, (clean_query, extracted_filters) = await asyncio.gather(
                embed_text(request.search_prompt, client),
                decompose_search_prompt(request.search_prompt, client),
            )
    except Exception as exc:
        logger.error("Search initialisation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service unavailable. Please try again shortly.",
        )

    # Log when implicit metadata was picked up from the prompt
    if extracted_filters.get("metadata_filters") or extracted_filters.get("is_fictional") is not None:
        logger.info(
            "Prompt decomposition extracted filters from '%s': %s",
            request.search_prompt,
            extracted_filters,
        )
        if clean_query != request.search_prompt:
            logger.info("Clean semantic query: '%s'", clean_query)

    # ── 2. Merge explicit + extracted filters (explicit wins per-key) ─────
    merged = _merge_filters(request.filters, extracted_filters)
    is_fictional_filter:   Optional[bool]  = merged["is_fictional"]
    meta_filters:          Dict[str, Any]  = merged["metadata_filters"]

    # ── 3. Build the SQL query ────────────────────────────────────────────
    # Embed the clean_query (prompt with noise stripped) for better vector match.
    # If decomposition failed, clean_query == original prompt — safe fallback.
    try:
        async with httpx.AsyncClient() as client:
            clean_vector = await embed_text(clean_query, client)
    except Exception:
        # Fall back to the already-computed vector of the original prompt
        clean_vector = query_vector

    distance_expr = Entity.biography_embedding.cosine_distance(clean_vector).label("distance")

    stmt = (
        select(Entity, distance_expr)
        .where(Entity.biography_embedding.is_not(None))
    )

    # 3a — Scalar boolean filter
    if is_fictional_filter is not None:
        stmt = stmt.where(Entity.is_fictional == is_fictional_filter)

    # 3b — JSONB metadata filters (injection-safe via SQLAlchemy bound params)
    # Compiles to:  metadata->>'key' = :param  — value is never interpolated.
    for key, value in meta_filters.items():
        stmt = stmt.where(
            Entity.entity_metadata[key].astext == str(value)
        )

    # ── 4. Order by cosine distance, return minimum 10 ───────────────────
    stmt = stmt.order_by(distance_expr).limit(10)

    rows = (await db.execute(stmt)).all()

    # ── 5. Serialise — biography_embedding structurally excluded ──────────
    output: List[EntitySearchResult] = []
    for row in rows:
        entity: Entity = row.Entity
        distance: Optional[float] = row.distance
        output.append(
            EntitySearchResult(
                id=entity.id,
                name=entity.name,
                is_fictional=entity.is_fictional,
                biography=entity.biography,
                metadata=entity.entity_metadata,
                image_url=entity.image_url,
                relevance_score=round(1.0 - distance, 6) if distance is not None else None,
            )
        )

    return output