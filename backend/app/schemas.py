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
    multiple_seals: Optional[list[Dict[str, Any]]] = None  # 多印鑑檢測結果（測試功能）
    
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


class SealInfo(BaseModel):
    """單個印鑑信息"""
    bbox: Dict[str, int] = Field(..., description="邊界框 {x, y, width, height}")
    center: Dict[str, Any] = Field(..., description="中心點 {center_x, center_y, radius}")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")


class MultipleSealsDetectionResponse(BaseModel):
    """多印鑑檢測響應模型"""
    detected: bool
    seals: list[SealInfo] = Field(default_factory=list, description="檢測到的印鑑列表")
    count: int = Field(0, description="檢測到的印鑑數量")
    reason: Optional[str] = Field(None, description="失敗原因")


class MultipleSealsSaveRequest(BaseModel):
    """保存多印鑑位置請求"""
    seals: list[SealInfo] = Field(..., description="印鑑列表")


class CropSealsRequest(BaseModel):
    """裁切印鑑請求"""
    seals: list[SealInfo] = Field(..., description="要裁切的印鑑列表")
    margin: Optional[int] = Field(15, ge=0, le=50, description="邊距（像素），默認15。用於確保裁切後的圖像邊緣有足夠的背景區域，供去背景函數檢測背景色")


class CropSealsResponse(BaseModel):
    """裁切印鑑響應"""
    cropped_image_ids: list[UUID] = Field(..., description="裁切後的圖像 ID 列表")
    count: int = Field(..., description="成功裁切的數量")


class MultiSealComparisonRequest(BaseModel):
    """多印鑑比對請求"""
    seal_image_ids: list[UUID] = Field(..., description="裁切後的印鑑圖像 ID 列表")
    threshold: Optional[float] = Field(0.95, ge=0.0, le=1.0, description="相似度閾值")
    # 傳統相似度權重參數（保留以向後兼容，但不再使用）
    similarity_ssim_weight: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="SSIM 權重")
    similarity_template_weight: Optional[float] = Field(0.35, ge=0.0, le=1.0, description="Template Match 權重")
    pixel_similarity_weight: Optional[float] = Field(0.1, ge=0.0, le=1.0, description="Pixel Similarity 權重")
    histogram_similarity_weight: Optional[float] = Field(0.05, ge=0.0, le=1.0, description="Histogram Similarity 權重")
    # Mask相似度權重參數
    overlap_weight: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="重疊區域權重")
    pixel_diff_penalty_weight: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="像素差異懲罰權重")
    unique_region_penalty_weight: Optional[float] = Field(0.2, ge=0.0, le=1.0, description="獨有區域懲罰權重")
    # 圖像對齊參數
    rotation_range: Optional[float] = Field(15.0, ge=0.0, le=180.0, description="旋轉角度搜索範圍（度）")
    translation_range: Optional[int] = Field(100, ge=0, le=1000, description="平移偏移搜索範圍（像素）")


class SealComparisonResult(BaseModel):
    """單個印鑑比對結果"""
    seal_index: int = Field(..., description="印鑑索引（從1開始）")
    seal_image_id: UUID = Field(..., description="印鑑圖像 ID")
    similarity: Optional[float] = Field(None, description="相似度")
    is_match: Optional[bool] = Field(None, description="是否匹配")
    overlay1_path: Optional[str] = Field(None, description="疊圖1路徑（圖像1疊在印鑑上）")
    overlay2_path: Optional[str] = Field(None, description="疊圖2路徑（印鑑疊在圖像1上）")
    heatmap_path: Optional[str] = Field(None, description="熱力圖路徑")
    overlap_mask_path: Optional[str] = Field(None, description="重疊區域mask路徑")
    pixel_diff_mask_path: Optional[str] = Field(None, description="像素差異mask路徑")
    diff_mask_2_only_path: Optional[str] = Field(None, description="圖像2獨有區域mask路徑")
    diff_mask_1_only_path: Optional[str] = Field(None, description="圖像1獨有區域mask路徑")
    gray_diff_path: Optional[str] = Field(None, description="灰度差異圖路徑（熱力圖視覺化）")
    mask_statistics: Optional[Dict[str, Any]] = Field(None, description="Mask統計資訊")
    mask_based_similarity: Optional[float] = Field(None, description="基於mask的相似度")
    input_image1_path: Optional[str] = Field(None, description="疊圖前的圖像1路徑（去背景後的裁切圖像）")
    input_image2_path: Optional[str] = Field(None, description="疊圖前的圖像2路徑（對齊後的印鑑圖像）")
    error: Optional[str] = Field(None, description="錯誤訊息（如果比對失敗）")
    overlay_error: Optional[str] = Field(None, description="疊圖生成錯誤訊息（如果疊圖生成失敗）")
    timing: Optional[Dict[str, Any]] = Field(None, description="時間追蹤數據（各步驟耗時，單位：秒）。可能包含嵌套字典 alignment_stages 記錄對齊各階段時間")


class MultiSealComparisonResponse(BaseModel):
    """多印鑑比對響應"""
    image1_id: UUID = Field(..., description="圖像1 ID")
    results: list[SealComparisonResult] = Field(..., description="比對結果列表")
    total_count: int = Field(..., description="總比對數量")
    success_count: int = Field(..., description="成功比對數量")


class MultiSealComparisonTaskCreate(BaseModel):
    """創建多印鑑比對任務請求"""
    task_uid: str = Field(..., description="任務 UID（用於追蹤）")
    image1_id: UUID = Field(..., description="圖像1 ID")
    seal_image_ids: list[UUID] = Field(..., description="裁切後的印鑑圖像 ID 列表")
    threshold: Optional[float] = Field(0.95, ge=0.0, le=1.0, description="相似度閾值")
    similarity_ssim_weight: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="SSIM 權重")
    similarity_template_weight: Optional[float] = Field(0.35, ge=0.0, le=1.0, description="Template Match 權重")
    pixel_similarity_weight: Optional[float] = Field(0.1, ge=0.0, le=1.0, description="Pixel Similarity 權重")
    histogram_similarity_weight: Optional[float] = Field(0.05, ge=0.0, le=1.0, description="Histogram Similarity 權重")


class MultiSealComparisonTaskResponse(BaseModel):
    """多印鑑比對任務響應"""
    id: UUID
    task_uid: str
    image1_id: UUID
    status: str
    progress: Optional[float] = None
    progress_message: Optional[str] = None
    results: Optional[list[SealComparisonResult]] = None
    total_count: Optional[int] = None
    success_count: Optional[int] = None
    error: Optional[str] = None
    error_trace: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    task_timing: Optional[Dict[str, float]] = Field(None, description="任務級別時間追蹤數據（總時間、平均時間等，單位：秒）")
    
    class Config:
        from_attributes = True


class MultiSealComparisonTaskStatusResponse(BaseModel):
    """多印鑑比對任務狀態響應"""
    task_uid: str
    status: str
    progress: Optional[float] = None
    progress_message: Optional[str] = None
    total_count: Optional[int] = None
    success_count: Optional[int] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class RotatedSealMatch(BaseModel):
    """旋轉匹配的印鑑結果"""
    bbox: Dict[str, int] = Field(..., description="邊界框 {x, y, width, height}")
    center: Dict[str, float] = Field(..., description="中心點 {center_x, center_y, radius}")
    rotation_angle: float = Field(..., description="最佳旋轉角度（度）")
    similarity: float = Field(..., ge=0.0, le=1.0, description="相似度分數")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")


class RotatedSealsDetectionResponse(BaseModel):
    """旋轉匹配的多印鑑偵測響應"""
    detected: bool = Field(..., description="是否檢測到匹配的印鑑")
    matches: list[RotatedSealMatch] = Field(default_factory=list, description="匹配的印鑑列表")
    count: int = Field(0, description="匹配的印鑑數量")
    reason: Optional[str] = Field(None, description="失敗原因")


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
        """從 details 中提取 translation_offset（優先從頂層，否則從 alignment_optimization）"""
        if self.details and isinstance(self.details, dict):
            # 優先從頂層獲取
            if 'translation_offset' in self.details and self.translation_offset is None:
                self.translation_offset = self.details.get('translation_offset')
            # 如果頂層沒有，從 alignment_optimization 獲取
            elif self.translation_offset is None:
                alignment_opt = self.details.get('alignment_optimization', {})
                if alignment_opt and 'translation_offset' in alignment_opt:
                    self.translation_offset = alignment_opt.get('translation_offset')
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

