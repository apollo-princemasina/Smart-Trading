# Reserved — Phase 3
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/sentiment", tags=["sentiment"])


@router.get("")
async def get_sentiment():
    raise HTTPException(503, detail="Sentiment endpoint not yet implemented (Phase 3)")
