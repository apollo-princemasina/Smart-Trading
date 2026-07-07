# Reserved — Phase 3
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/news", tags=["news"])


@router.get("")
async def get_news():
    raise HTTPException(503, detail="News endpoint not yet implemented (Phase 3)")
