"""Admin index tuning router: recommendations and application."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from config.database import get_db
from models.pydantic_models import IndexTuningResponse
from services.index_tuner import IndexTuner

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Admin"])


def get_tuner(db: Session = Depends(get_db)) -> IndexTuner:
    return IndexTuner(db)


@router.get("/admin/index-tuning/recommendations", response_model=IndexTuningResponse)
def get_recommendations(tuner: IndexTuner = Depends(get_tuner)):
    try:
        recs = tuner.get_recommendations()
        return IndexTuningResponse(success=True, recommendations=recs)
    except Exception as exc:
        logger.exception("Failed to get index tuning recommendations")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/admin/index-tuning/apply/{collection_id}")
def apply_recommendations(
    collection_id: str,
    tuner: IndexTuner = Depends(get_tuner),
):
    try:
        result = tuner.apply_recommendations(collection_id)
        return {"success": True, "result": result}
    except Exception as exc:
        logger.exception("Failed to apply index tuning")
        raise HTTPException(status_code=500, detail=str(exc))
