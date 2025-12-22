"""
統計相關 API
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.schemas import StatisticsResponse, ComparisonResponse
from app.models import Comparison

router = APIRouter(prefix="/statistics", tags=["statistics"])


@router.get("/", response_model=StatisticsResponse)
def get_statistics(
    db: Session = Depends(get_db)
):
    """
    獲取統計資訊
    """
    # 基礎查詢：只包含未刪除的記錄
    base_query = db.query(Comparison).filter(Comparison.deleted_at.is_(None))
    
    # 總比對次數
    total_comparisons = base_query.count()
    
    # 匹配次數
    match_count = base_query.filter(Comparison.is_match == True).count()
    
    # 不匹配次數
    mismatch_count = base_query.filter(Comparison.is_match == False).count()
    
    # 平均相似度
    avg_similarity_result = base_query.filter(
        Comparison.similarity.isnot(None)
    ).with_entities(func.avg(Comparison.similarity)).scalar()
    average_similarity = float(avg_similarity_result) if avg_similarity_result else 0.0
    
    # 最近的比對記錄（最近10條）
    recent_comparisons = base_query.order_by(
        Comparison.created_at.desc()
    ).limit(10).all()
    
    return StatisticsResponse(
        total_comparisons=total_comparisons,
        match_count=match_count,
        mismatch_count=mismatch_count,
        average_similarity=average_similarity,
        recent_comparisons=[ComparisonResponse.model_validate(c) for c in recent_comparisons]
    )

