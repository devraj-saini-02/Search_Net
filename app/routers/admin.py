"""
app/routers/admin.py
--------------------
Admin-only endpoint: ingestion queue.

All routes in this router are protected by verify_admin (HTTP Basic Auth).
A 202 is returned immediately; the ETL pipeline runs in the background.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, status

from app.auth import verify_admin
from app.etl_worker import run_etl_pipeline
from app.schemas import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["Admin"])


@router.post(
    "/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestResponse,
    summary="Queue an async ETL ingestion batch",
    description=(
        "Accepts a batch of name + context_hint pairs and returns **202 Accepted** "
        "immediately. The full ETL pipeline (Wikipedia → Llama-3 → Embedder → "
        "TMDB → PostgreSQL upsert) runs asynchronously in the background.\n\n"
        "**Access:** Admin Basic Auth only (`admin_architect` / `unbreakable_secure_hash`)."
    ),
)
async def ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(verify_admin),
) -> IngestResponse:
    background_tasks.add_task(
        run_etl_pipeline,
        items=request.items,
        is_fictional=request.is_fictional,
    )
    logger.info(
        "Ingestion queued — %d item(s), is_fictional=%s",
        len(request.items),
        request.is_fictional,
    )
    return IngestResponse(
        message="Ingestion pipeline queued successfully.",
        item_count=len(request.items),
    )