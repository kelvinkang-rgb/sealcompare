"""
視覺化相關 API
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from uuid import UUID
from pathlib import Path

from app.database import get_db
from app.models import VisualizationType
from app.services.comparison_service import ComparisonService

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
    comparison_service = ComparisonService(db)
    vis = comparison_service.get_comparison_visualization(
        comparison_id, 
        VisualizationType.COMPARISON_IMAGE
    )
    
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
    comparison_service = ComparisonService(db)
    vis = comparison_service.get_comparison_visualization(
        comparison_id,
        VisualizationType.HEATMAP
    )
    
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
    overlay_type: str = Query("1", description="疊圖類型（'1' 或 '2'）"),
    db: Session = Depends(get_db)
):
    """
    獲取疊圖
    
    - **comparison_id**: 比對 ID
    - **overlay_type**: 疊圖類型（"1" 或 "2"）
    """
    vis_type = VisualizationType.OVERLAY1 if overlay_type == "1" else VisualizationType.OVERLAY2
    
    comparison_service = ComparisonService(db)
    vis = comparison_service.get_comparison_visualization(
        comparison_id,
        vis_type
    )
    
    file_path = Path(vis.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="疊圖文件不存在")
    
    return FileResponse(
        path=file_path,
        media_type="image/png",
        filename=f"overlay{overlay_type}_{comparison_id}.png"
    )

