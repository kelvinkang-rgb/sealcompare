"""
統計相關 API
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas import StatisticsResponse, ComparisonResponse
from app.services.comparison_service import ComparisonService

router = APIRouter(prefix="/statistics", tags=["statistics"])


@router.get("/", response_model=StatisticsResponse)
def get_statistics(
    db: Session = Depends(get_db)
):
    """
    獲取統計資訊
    """
    comparison_service = ComparisonService(db)
    stats = comparison_service.get_statistics()
    
    return StatisticsResponse(
        total_comparisons=stats['total_comparisons'],
        match_count=stats['match_count'],
        mismatch_count=stats['mismatch_count'],
        average_similarity=stats['average_similarity'],
        recent_comparisons=[ComparisonResponse.model_validate(c) for c in stats['recent_comparisons']]
    )

