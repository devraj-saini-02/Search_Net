"""
app/etl_worker.py
-----------------
Asynchronous ETL pipeline invoked via FastAPI BackgroundTasks.

Per-entity pipeline (6 steps):
  1. fetch_wikipedia_bio()          Wikipedia search + raw intro extract
  2. truncate_text()                Hard-cap raw context before the LLM sees it
  3. extract_entity_data_with_llm() ONE LLM call → {"summary": "...", "metadata": {...}}
                                    summary  = 2-3 sentence blurb stored as biography
                                    metadata = structured JSONB tree
  4. embed_text()                   all-MiniLM-L6-v2 → 384-dim vector of the summary
  5. fetch_tmdb_image()             TMDB /search/multi → poster URL (fictional only)
  6. upsert_entity()                PostgreSQL INSERT … ON CONFLICT DO UPDATE

Also exposes decompose_search_prompt() for search-time implicit filter extraction.

Error handling (Option A — never abort the batch):
  - Wikipedia miss  → fallback context = "{name}: {context_hint}",  continue
  - LLM failure     → summary = fallback text, metadata = {},        continue
  - TMDB failure    → image_url = None,                              continue
  - Embedding fail  → log error, SKIP entity entirely (vector is mandatory)
"""

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.database import AsyncSessionLocal
from app.models import Entity
from app.schemas import IngestItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Characters of raw Wikipedia text sent to the LLM as context.
# The LLM uses this to write a concise summary — the raw text is then discarded.
# Keeping this tight prevents token-limit crashes on the HF free tier.
MAX_CONTEXT_CHARS: int = 1_500

HF_LLM_URL = (
    "https://api-inference.huggingface.co/models/"
    "meta-llama/Meta-Llama-3-8B-Instruct"
)
HF_EMBED_URL = (
    "https://api-inference.huggingface.co/models/"
    "sentence-transformers/all-MiniLM-L6-v2"
)

WIKIPEDIA_API   = "https://en.wikipedia.org/w/api.php"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/multi"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def truncate_text(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Hard-cap raw text to max_chars for LLM context.
    Cuts at the last sentence boundary if one exists in the back half;
    otherwise cuts cleanly with an ellipsis marker.
    This text is ONLY used as LLM context — it is never stored directly.
    """
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    last_period = window.rfind(".")
    if last_period > max_chars // 2:
        return window[: last_period + 1]
    return window.rstrip() + "…"


def _hf_headers() -> Dict[str, str]:
    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is not set in environment / .env file.")
    return {"Authorization": f"Bearer {token}"}


async def _hf_post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    max_retries: int = 3,
    backoff: float = 5.0,
) -> httpx.Response:
    """
    POST to a HF Inference endpoint with retry + exponential backoff.

    Retries on:
      - ConnectError / DNS failures  ([Errno 11001] getaddrinfo failed on Windows)
      - HTTP 503 (model loading / cold start — very common on free HF tier)
      - HTTP 429 (rate limited)

    Raises the final exception if all retries are exhausted.
    """
    import asyncio

    last_exc: Exception = RuntimeError("No attempts made.")
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.post(
                url, json=payload, headers=_hf_headers(), timeout=90.0
            )
            # Retry on HF cold-start (503) and rate-limit (429)
            if resp.status_code in (429, 503):
                wait = backoff * attempt
                logger.warning(
                    "HF API returned %d on attempt %d — retrying in %.0fs …",
                    resp.status_code, attempt, wait,
                )
                await asyncio.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            wait = backoff * attempt
            logger.warning(
                "HF network error on attempt %d/%d: %s — retrying in %.0fs …",
                attempt, max_retries, exc, wait,
            )
            last_exc = exc
            await asyncio.sleep(wait)

    raise last_exc


def _llm_call_payload(prompt: str, max_new_tokens: int = 512) -> Dict[str, Any]:
    """Shared Llama-3 inference payload factory."""
    return {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.05,          # Near-deterministic for structured output
            "return_full_text": False,
            "stop": ["<|eot_id|>"],
        },
    }


def _build_llama_prompt(system: str, user: str) -> str:
    """Wraps system + user messages in the Llama-3 chat template."""
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def _extract_json(raw: str) -> Dict[str, Any]:
    """
    Robustly pulls the first complete JSON object out of a raw LLM string.
    Handles cases where the model emits stray text before or after the braces.
    """
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in LLM output.")
    return json.loads(raw[start:end])


# ---------------------------------------------------------------------------
# Step 1 — Wikipedia biography
# ---------------------------------------------------------------------------

async def fetch_wikipedia_bio(
    name: str, context_hint: str, client: httpx.AsyncClient
) -> str:
    """
    Two-step Wikipedia lookup:
      a) Opensearch with "{name} {context_hint}" → resolves the best page title.
      b) extracts action → returns the plain-text intro paragraph.

    Raises ValueError on misses so the caller can apply the fallback strategy.
    """
    query = f"{name} {context_hint}"

    # a) Resolve page title via search
    search_resp = await client.get(
        WIKIPEDIA_API,
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
            "format": "json",
        },
        timeout=12.0,
    )
    search_resp.raise_for_status()
    hits = search_resp.json().get("query", {}).get("search", [])
    if not hits:
        raise ValueError(f"No Wikipedia results for: '{query}'")

    page_title: str = hits[0]["title"]
    logger.debug("Wikipedia resolved '%s' → '%s'", name, page_title)

    # b) Fetch plain-text intro extract
    extract_resp = await client.get(
        WIKIPEDIA_API,
        params={
            "action": "query",
            "prop": "extracts",
            "exintro": True,       # Intro section only — no full article
            "explaintext": True,   # Strip all wiki markup
            "titles": page_title,
            "format": "json",
        },
        timeout=12.0,
    )
    extract_resp.raise_for_status()
    pages = extract_resp.json().get("query", {}).get("pages", {})
    extract: str = next(iter(pages.values())).get("extract", "").strip()

    if not extract:
        raise ValueError(f"Empty Wikipedia extract for: '{page_title}'")
    return extract


# ---------------------------------------------------------------------------
# Step 3 — Single LLM call: summary + metadata extraction
# ---------------------------------------------------------------------------

async def extract_entity_data_with_llm(
    name: str,
    raw_context: str,
    is_fictional: bool,
    client: httpx.AsyncClient,
) -> Tuple[str, Dict[str, Any]]:
    """
    ONE Llama-3-8B call that returns both:
      - summary:  2-3 sentence plain-English blurb stored as biography.
                  Replaces the raw Wikipedia wall-of-text with something
                  concise and useful for a "find a person" tool.
      - metadata: structured JSONB tree (profession/universe/etc.)

    Combining both into a single LLM call halves HF API usage vs. the
    previous approach of separate extraction + summarisation calls.

    Returns ("fallback summary", {}) on any LLM or parse failure so the
    pipeline can continue without blocking the upsert.
    """
    if is_fictional:
        meta_schema = (
            '"universe": "franchise/universe name or null", '
            '"creators": ["list of creator names"], '
            '"first_appearance": "string or null", '
            '"abilities": ["key abilities or powers"], '
            '"affiliation": "main team or organisation or null"'
        )
    else:
        meta_schema = (
            '"profession": "primary job or title or null", '
            '"nationality": "country of origin or null", '
            '"timeline": {"born": year_integer_or_null, "died": year_integer_or_null}, '
            '"known_for": ["2-3 major achievements"], '
            '"field": "primary field of work or null"'
        )

    entity_type = "fictional character" if is_fictional else "real person"

    system = (
        "You are a data extraction engine. "
        "Your ONLY output is a single raw JSON object — "
        "no markdown fences, no explanation, no text outside the braces."
    )

    user = f"""You are processing the {entity_type} named "{name}".

Using the biography context below, return a JSON object with EXACTLY these two keys:

{{
  "summary": "Write 2-3 concise sentences: who they are, what they are most known for, and one standout fact. Plain English. No lists.",
  "metadata": {{ {meta_schema} }}
}}

Biography context (use only as reference — do not copy verbatim):
{raw_context}"""

    prompt = _build_llama_prompt(system, user)

    fallback_summary = f"{name} — {('fictional character' if is_fictional else 'notable individual')}."

    try:
        resp = await _hf_post_with_retry(
            client, HF_LLM_URL, _llm_call_payload(prompt, max_new_tokens=512)
        )
        raw: str = resp.json()[0]["generated_text"].strip()
        parsed = _extract_json(raw)

        summary  = str(parsed.get("summary", fallback_summary)).strip()
        metadata = parsed.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        return summary, metadata

    except Exception as exc:
        logger.warning("LLM entity data extraction failed for '%s': %s", name, exc)
        return fallback_summary, {}


# ---------------------------------------------------------------------------
# Step 4 — Text embedder (all-MiniLM-L6-v2 via HF Serverless)
# ---------------------------------------------------------------------------

async def embed_text(text: str, client: httpx.AsyncClient) -> List[float]:
    """
    Returns a 384-dimensional float vector for the supplied text.

    HF feature-extraction returns shape [[384 floats]] for a single input;
    the outer batch dimension is unwrapped automatically.

    Raises on failure — embeddings are mandatory for an entity to be stored.
    """
    resp = await _hf_post_with_retry(
        client, HF_EMBED_URL, {"inputs": text}
    )
    result = resp.json()

    # Unwrap batch dimension: [[f1, f2, …]] → [f1, f2, …]
    if isinstance(result, list) and result and isinstance(result[0], list):
        return result[0]
    return result


# ---------------------------------------------------------------------------
# Step 5 — TMDB image (fictional entities only)
# ---------------------------------------------------------------------------

async def fetch_tmdb_image(
    name: str, context_hint: str, client: httpx.AsyncClient
) -> Optional[str]:
    """
    Searches TMDB /search/multi for the franchise / film matching the character.
    Returns the first available poster URL, or None if nothing is found.
    """
    api_key = os.getenv("TMDB_API_KEY", "")
    if not api_key:
        logger.warning("TMDB_API_KEY not set — skipping image fetch for '%s'", name)
        return None

    try:
        resp = await client.get(
            TMDB_SEARCH_URL,
            params={
                "api_key": api_key,
                "query": f"{name} {context_hint}",
                "include_adult": "false",
                "page": 1,
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        for item in resp.json().get("results", []):
            path = item.get("poster_path") or item.get("profile_path")
            if path:
                return f"{TMDB_IMAGE_BASE}{path}"
    except Exception as exc:
        logger.warning("TMDB image fetch failed for '%s': %s", name, exc)

    return None


# ---------------------------------------------------------------------------
# Step 6 — PostgreSQL atomic upsert
# ---------------------------------------------------------------------------

async def upsert_entity(
    name: str,
    is_fictional: bool,
    biography: str,
    embedding: List[float],
    meta: Dict[str, Any],
    image_url: Optional[str],
) -> None:
    """
    INSERT … ON CONFLICT (name, is_fictional) DO UPDATE.
    Fully atomic — safe for concurrent ETL workers.
    """
    async with AsyncSessionLocal() as session:
        stmt = (
            pg_insert(Entity)
            .values(
                id=uuid.uuid4(),
                name=name,
                is_fictional=is_fictional,
                biography=biography,
                biography_embedding=embedding,
                metadata=meta,       # DB column name — maps to entity_metadata attr
                image_url=image_url,
            )
            .on_conflict_do_update(
                constraint="uq_entity_name_fictional",
                set_={
                    "biography": biography,
                    "metadata": meta,
                    "biography_embedding": embedding,
                    "image_url": image_url,
                },
            )
        )
        await session.execute(stmt)
        await session.commit()
    logger.info("Upserted entity: '%s' (fictional=%s)", name, is_fictional)


# ---------------------------------------------------------------------------
# Search-time: implicit filter extraction from natural language prompt
# ---------------------------------------------------------------------------

async def decompose_search_prompt(
    prompt: str, client: httpx.AsyncClient
) -> Tuple[str, Dict[str, Any]]:
    """
    Splits a natural language search query into:
      clean_query       — descriptive core terms, stripped of entity-type /
                          universe / franchise words that belong in filters.
      extracted_filters — {"is_fictional": bool|None, "metadata_filters": {...}}

    Called concurrently with embed_text() in the search endpoint via
    asyncio.gather() so it adds zero wall-clock latency to the search path.

    Example:
      Input:  "Marvel genius inventor superhero"
      Output: ("genius inventor", {"is_fictional": True, "metadata_filters": {"universe": "Marvel"}})

    Returns the original prompt + empty filters on any LLM/parse failure —
    the search gracefully degrades to pure semantic matching.
    """
    system = (
        "You decompose search queries about people into structured search intent. "
        "Your ONLY output is a single raw JSON object — no markdown, no explanation."
    )

    user = f"""Decompose this search query: "{prompt}"

Return ONLY this JSON:
{{
  "clean_query": "descriptive core terms only — remove universe, franchise, and entity-type words",
  "is_fictional": true or false or null,
  "metadata_filters": {{}}
}}

Rules for metadata_filters — include a key ONLY if it is clearly and explicitly mentioned:
  "universe"     → franchise or fictional universe  (e.g. Marvel, DC, Star Wars)
  "profession"   → job or role of a real person     (e.g. Physicist, Politician)
  "field"        → domain of work                   (e.g. Physics, Music)
  "nationality"  → country                          (e.g. German, American)
  "affiliation"  → team or organisation             (e.g. Avengers, NASA)

If nothing qualifies, return an empty object for metadata_filters."""

    prompt_str = _build_llama_prompt(system, user)
    fallback = (prompt, {"is_fictional": None, "metadata_filters": {}})

    try:
        resp = await _hf_post_with_retry(
            client, HF_LLM_URL, _llm_call_payload(prompt_str, max_new_tokens=200)
        )
        raw: str = resp.json()[0]["generated_text"].strip()
        parsed = _extract_json(raw)

        clean_query: str = str(parsed.get("clean_query", prompt)).strip() or prompt
        is_fictional = parsed.get("is_fictional")          # may be bool or None
        meta_filters: Dict[str, Any] = parsed.get("metadata_filters", {})

        if not isinstance(meta_filters, dict):
            meta_filters = {}

        extracted_filters = {
            "is_fictional": is_fictional,
            "metadata_filters": meta_filters,
        }
        return clean_query, extracted_filters

    except Exception as exc:
        logger.warning("Prompt decomposition failed for '%s': %s — using raw prompt.", prompt, exc)
        return fallback


# ---------------------------------------------------------------------------
# Internal — single entity orchestrator
# ---------------------------------------------------------------------------

async def _process_single_entity(
    item: IngestItem, is_fictional: bool, client: httpx.AsyncClient
) -> None:
    name, hint = item.name, item.context_hint

    # Step 1 — Fetch raw Wikipedia context (with graceful fallback)
    try:
        raw_context = await fetch_wikipedia_bio(name, hint, client)
    except Exception as exc:
        logger.warning("Wikipedia miss for '%s': %s — using fallback context.", name, exc)
        raw_context = f"{name}: {hint}"

    # Step 2 — Truncate raw context before LLM (guards against token-limit crashes)
    llm_context = truncate_text(raw_context)

    # Step 3 — Single LLM call: concise summary bio + structured metadata
    #   summary  → stored as biography (2-3 sentences, not a wall of text)
    #   metadata → stored as JSONB
    summary, meta = await extract_entity_data_with_llm(
        name, llm_context, is_fictional, client
    )

    # Step 4 — Embed the concise summary (not the raw Wikipedia dump)
    #   The embedding now represents a clean, dense representation of the person.
    embedding = await embed_text(summary, client)   # raises → skips entity

    # Step 5 — TMDB poster image (fictional entities only)
    image_url: Optional[str] = None
    if is_fictional:
        image_url = await fetch_tmdb_image(name, hint, client)

    # Step 6 — Atomic upsert into PostgreSQL
    await upsert_entity(name, is_fictional, summary, embedding, meta, image_url)


# ---------------------------------------------------------------------------
# Public entry point — called by BackgroundTasks in admin router
# ---------------------------------------------------------------------------

async def run_etl_pipeline(items: List[IngestItem], is_fictional: bool) -> None:
    """
    Processes a batch of IngestItems sequentially.
    Single shared httpx.AsyncClient for connection pooling across the batch.
    Per-item exceptions are caught and logged — one bad row never aborts the batch.
    """
    logger.info(
        "ETL pipeline started — %d item(s), is_fictional=%s", len(items), is_fictional
    )
    async with httpx.AsyncClient(
        follow_redirects=True,
        headers={
            # Wikipedia requires a descriptive User-Agent with a contact method.
            # A generic string like "python-httpx" gets 403'd automatically.
            "User-Agent": (
                "SearchEngine-ETL/1.0 "
                "(https://github.com/devraj-saini-02/Search_Net; devraj.saini.ug23@nsut.ac.in) "
                "python-httpx"
            )
        },
    ) as client:
        for item in items:
            try:
                await _process_single_entity(item, is_fictional, client)
            except Exception as exc:
                logger.error(
                    "ETL failed for '%s' (fictional=%s): %s",
                    item.name, is_fictional, exc, exc_info=True,
                )
    logger.info("ETL pipeline finished — batch of %d item(s).", len(items))
