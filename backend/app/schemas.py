"""
Pydantic 數據模型（用於 API 請求和響應）
"""

from pydantic import BaseModel, Field, field_validator, model_validator
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
    seal_detected: bool = False
    seal_confidence: Optional[float] = None
    seal_bbox: Optional[Dict[str, int]] = None
    seal_center: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


class SealDetectionResponse(BaseModel):
    """印鑑檢測響應模型"""
    detected: bool
    confidence: float
    bbox: Optional[Dict[str, int]] = None
    center: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


class SealLocationUpdate(BaseModel):
    """更新印鑑位置請求"""
    bbox: Optional[Dict[str, int]] = Field(None, description="邊界框 {x, y, width, height}")
    center: Optional[Dict[str, Any]] = Field(None, description="中心點 {center_x, center_y, radius}")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="置信度")


# 比對相關 Schema
class ComparisonBase(BaseModel):
    """比對基礎模型"""
    image1_id: UUID
    image2_id: UUID
    threshold: float = Field(default=0.95, ge=0.0, le=1.0)


class ComparisonCreate(ComparisonBase):
    """創建比對請求"""
    enable_rotation_search: bool = True
    enable_translation_search: bool = True  # 預設開啟，因為人工標記印鑑無法確保中心點都一致


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
    translation_offset: Optional[Dict[str, int]] = None  # {"x": int, "y": int}
    similarity_before_correction: Optional[float] = None
    improvement: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    center_offset: Optional[float] = None
    size_ratio: Optional[float] = None
    notes: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    deleted_at: Optional[datetime] = None
    
    @model_validator(mode='after')
    def extract_translation_offset(self):
        """從 details 中提取 translation_offset"""
        if self.details and isinstance(self.details, dict) and 'translation_offset' in self.details:
            if self.translation_offset is None:
                self.translation_offset = self.details.get('translation_offset')
        return self
    
    class Config:
        from_attributes = True
        use_enum_values = True


class ComparisonStatusResponse(BaseModel):
    """比對狀態響應"""
    id: UUID
    status: ComparisonStatus
    progress: Optional[float] = None  # 0-100
    message: Optional[str] = None


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

