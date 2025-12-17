"""
視覺化相關 API
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from uuid import UUID
from pathlib import Path

from app.database import get_db
from app.models import Comparison, ComparisonVisualization, VisualizationType

router = APIRouter(prefix="/comparisons", tags=["visualizations"])


@router.get("/{comparison_id}/comparison-image")
def get_comparison_image(
    comparison_id: UUID,
    db: Session = Depends(get_db)
):
    """
    獲取並排對比圖
    
    - **comparison_id**: 比對 ID
    """
    comparison = db.query(Comparison).filter(Comparison.id == comparison_id).first()
    if not comparison:
        raise HTTPException(status_code=404, detail="比對記錄不存在")
    
    vis = db.query(ComparisonVisualization).filter(
        ComparisonVisualization.comparison_id == comparison_id,
        ComparisonVisualization.type == VisualizationType.COMPARISON_IMAGE
    ).first()
    
    if not vis:
        raise HTTPException(status_code=404, detail="對比圖不存在")
    
    file_path = Path(vis.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="對比圖文件不存在")
    
    return FileResponse(
        path=file_path,
        media_type="image/jpeg",
        filename=f"comparison_{comparison_id}.jpg"
    )


@router.get("/{comparison_id}/heatmap")
def get_heatmap(
    comparison_id: UUID,
    db: Session = Depends(get_db)
):
    """
    獲取差異熱力圖
    
    - **comparison_id**: 比對 ID
    """
    comparison = db.query(Comparison).filter(Comparison.id == comparison_id).first()
    if not comparison:
        raise HTTPException(status_code=404, detail="比對記錄不存在")
    
    vis = db.query(ComparisonVisualization).filter(
        ComparisonVisualization.comparison_id == comparison_id,
        ComparisonVisualization.type == VisualizationType.HEATMAP
    ).first()
    
    if not vis:
        raise HTTPException(status_code=404, detail="熱力圖不存在")
    
    file_path = Path(vis.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="熱力圖文件不存在")
    
    return FileResponse(
        path=file_path,
        media_type="image/jpeg",
        filename=f"heatmap_{comparison_id}.jpg"
    )


@router.get("/{comparison_id}/overlay")
def get_overlay(
    comparison_id: UUID,
    overlay_type: str = "1",  # "1" 或 "2"
    db: Session = Depends(get_db)
):
    """
    獲取疊圖
    
    - **comparison_id**: 比對 ID
    - **overlay_type**: 疊圖類型（"1" 或 "2"）
    """
    comparison = db.query(Comparison).filter(Comparison.id == comparison_id).first()
    if not comparison:
        raise HTTPException(status_code=404, detail="比對記錄不存在")
    
    vis_type = VisualizationType.OVERLAY1 if overlay_type == "1" else VisualizationType.OVERLAY2
    
    vis = db.query(ComparisonVisualization).filter(
        ComparisonVisualization.comparison_id == comparison_id,
        ComparisonVisualization.type == vis_type
    ).first()
    
    if not vis:
        raise HTTPException(status_code=404, detail="疊圖不存在")
    
    file_path = Path(vis.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="疊圖文件不存在")
    
    return FileResponse(
        path=file_path,
        media_type="image/png",
        filename=f"overlay{overlay_type}_{comparison_id}.png"
    )

