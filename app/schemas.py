"""
app/schemas.py
--------------
Pydantic v2 request and response models for every endpoint.
"""

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class IngestItem(BaseModel):
    """One name + context-hint pair submitted to the ETL pipeline."""

    name: str = Field(..., min_length=1, max_length=255, examples=["Albert Einstein"])
    context_hint: str = Field(
        ...,
        min_length=1,
        examples=["theoretical physicist, general relativity"],
        description=(
            "Free-text disambiguation hint. Used to select the correct Wikipedia "
            "page and guide LLM metadata extraction. "
            "E.g. 'physicist quantum mechanics' or 'DC Comics villain Gotham City'."
        ),
    )


class IngestRequest(BaseModel):
    """Admin payload for POST /api/v1/admin/ingest."""

    is_fictional: bool = Field(
        ...,
        description=(
            "True  → fictional character (TMDB image lookup enabled). "
            "False → real person (Wikipedia bio only)."
        ),
    )
    items: List[IngestItem] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    message: str
    item_count: int


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class SearchFilters(BaseModel):
    """
    Optional filter block attached to the hybrid search request.

    is_fictional       → scalar boolean filter on entities.is_fictional.
    metadata_filters   → freeform key/value JSONB equality checks.
                         All values coerced to str for the ->> comparison.
                         Example: {"universe": "Marvel", "field": "Physics"}
    """

    is_fictional: Optional[bool] = None
    metadata_filters: Optional[Dict[str, Any]] = Field(
        default=None,
        examples=[{"universe": "Marvel", "affiliation": "Avengers"}],
    )


class SearchRequest(BaseModel):
    """Payload for POST /api/v1/search."""

    search_prompt: str = Field(
        ...,
        min_length=1,
        examples=["eccentric geniuses who changed the world"],
    )
    filters: Optional[SearchFilters] = None


# ---------------------------------------------------------------------------
# Autocomplete
# ---------------------------------------------------------------------------

class SuggestResult(BaseModel):
    name: str
    image_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Search results  —  biography_embedding intentionally absent
# ---------------------------------------------------------------------------

class EntitySearchResult(BaseModel):
    """
    Single item in the hybrid search response.
    biography_embedding is structurally excluded — it never appears in output.
    """

    id: uuid.UUID
    name: str
    is_fictional: bool
    biography: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    image_url: Optional[str] = None
    relevance_score: Optional[float] = Field(
        default=None,
        description="1 − cosine_distance. Range [0, 1]. Higher = more relevant.",
    )