"""
Pydantic 數據模型（用於 API 請求和響應）
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from app.models import ComparisonStatus, VisualizationType


# 圖像相關 Schema
class ImageBase(BaseModel):
    """圖像基礎模型"""
    filename: str
    file_path: str
    file_size: Optional[str] = None
    mime_type: Optional[str] = None


class ImageCreate(ImageBase):
    """創建圖像請求"""
    pass


class ImageResponse(ImageBase):
    """圖像響應模型"""
    id: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# 比對相關 Schema
class ComparisonBase(BaseModel):
    """比對基礎模型"""
    image1_id: UUID
    image2_id: UUID
    threshold: float = Field(default=0.95, ge=0.0, le=1.0)


class ComparisonCreate(ComparisonBase):
    """創建比對請求"""
    enable_rotation_search: bool = True


class ComparisonUpdate(BaseModel):
    """更新比對請求"""
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="相似度閾值")
    notes: Optional[str] = Field(None, max_length=500, description="備註或標籤")
    status: Optional[ComparisonStatus] = Field(None, description="狀態（手動標記）")


class ComparisonResponse(ComparisonBase):
    """比對響應模型"""
    id: UUID
    image2_corrected_path: Optional[str] = None
    status: ComparisonStatus
    is_match: Optional[bool] = None
    similarity: Optional[float] = None
    rotation_angle: Optional[float] = None
    similarity_before_correction: Optional[float] = None
    improvement: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    center_offset: Optional[float] = None
    size_ratio: Optional[float] = None
    notes: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True
        use_enum_values = True


class ComparisonStatusResponse(BaseModel):
    """比對狀態響應"""
    id: UUID
    status: ComparisonStatus
    progress: Optional[float] = None  # 0-100
    message: Optional[str] = None
    current_step: Optional[str] = None  # 當前處理步驟


# 視覺化相關 Schema
class VisualizationResponse(BaseModel):
    """視覺化響應模型"""
    id: UUID
    comparison_id: UUID
    type: VisualizationType
    file_path: str
    created_at: datetime
    
    class Config:
        from_attributes = True


# 統計相關 Schema
class StatisticsResponse(BaseModel):
    """統計資訊響應"""
    total_comparisons: int
    match_count: int
    mismatch_count: int
    average_similarity: float
    recent_comparisons: list[ComparisonResponse]


# 報告相關 Schema
class ReportResponse(BaseModel):
    """報告響應模型"""
    comparison_id: UUID
    html_content: Optional[str] = None
    json_data: Optional[Dict[str, Any]] = None

