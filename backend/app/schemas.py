"""
Pydantic 數據模型（用於 API 請求和響應）
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from app.models import ComparisonStatus


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
    # PDF 相關（B：PDF 任務 + 分頁）
    is_pdf: bool = False
    pdf_page_count: Optional[int] = None
    source_pdf_id: Optional[UUID] = None
    page_index: Optional[int] = None
    pages: Optional[list["PdfPageInfo"]] = None
    
    class Config:
        from_attributes = True


class PdfPageInfo(BaseModel):
    """PDF 分頁圖像資訊（用於在 UI 展開頁列表）"""
    id: UUID
    page_index: int
    filename: str
    file_path: str
    mime_type: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class SealDetectionResponse(BaseModel):
    """印鑑檢測響應模型"""
    detected: bool
    confidence: float
    bbox: Optional[Dict[str, int]] = None
    center: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    # PDF：若 image_id 是 PDF，本次最佳偵測所在頁
    page_image_id: Optional[UUID] = None
    page_index: Optional[int] = None


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
    # PDF：逐頁結果（B 模式）
    pages: Optional[list["PdfPageSealsResult"]] = Field(default=None, description="PDF 每頁的偵測結果")
    total_count: Optional[int] = Field(default=None, description="PDF 全部頁總數量（count 的總和）")


class PdfPageSealsResult(BaseModel):
    """PDF 單頁多印鑑偵測結果"""
    page_index: int
    page_image_id: UUID
    detected: bool
    seals: list[SealInfo] = Field(default_factory=list)
    count: int = 0
    reason: Optional[str] = None


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
    threshold: Optional[float] = Field(0.83, ge=0.0, le=1.0, description="相似度閾值")
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
    structure_similarity: Optional[float] = Field(None, description="結構相似度（對印泥深淺較不敏感，0-1）")
    alignment_metrics: Optional[Dict[str, Any]] = Field(None, description="對齊過程指標（角度/偏移/救援/符號判別等）")
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
    threshold: Optional[float] = Field(0.83, ge=0.0, le=1.0, description="相似度閾值")
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


# ========= PDF 比對（B 模式：PDF 任務 + 分頁） =========
class PdfCompareRequest(BaseModel):
    """PDF 比對請求：image1 第1頁作模板，比對 image2 全部頁"""
    image2_pdf_id: UUID = Field(..., description="圖像2 的 PDF Image ID")
    max_seals: int = Field(160, ge=1, le=160, description="每頁最多偵測印鑑數")
    margin: int = Field(10, ge=0, le=50, description="裁切邊距（像素）")
    threshold: Optional[float] = Field(0.83, ge=0.0, le=1.0, description="相似度閾值")
    overlap_weight: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    pixel_diff_penalty_weight: Optional[float] = Field(0.3, ge=0.0, le=1.0)
    unique_region_penalty_weight: Optional[float] = Field(0.2, ge=0.0, le=1.0)
    rotation_range: Optional[float] = Field(15.0, ge=0.0, le=180.0)
    translation_range: Optional[int] = Field(100, ge=0, le=1000)


class PdfCompareTaskResponse(BaseModel):
    task_uid: str
    image1_id: UUID
    image2_pdf_id: UUID
    status: str
    progress: float = 0.0
    progress_message: Optional[str] = None
    pages_total: Optional[int] = None
    created_at: Optional[datetime] = None


class PdfComparePageResult(BaseModel):
    page_index: int
    page_image_id: UUID
    detected: bool
    count: int
    results: list[Dict[str, Any]] = Field(default_factory=list)
    reason: Optional[str] = None


class PdfCompareTaskStatusResponse(BaseModel):
    task_uid: str
    status: str
    progress: float = 0.0
    progress_message: Optional[str] = None
    pages_total: Optional[int] = None
    pages_done: Optional[int] = None


class PdfCompareTaskResultResponse(BaseModel):
    task_uid: str
    status: str
    pages_total: Optional[int] = None
    pages_done: Optional[int] = None
    results_by_page: list[PdfComparePageResult] = Field(default_factory=list)
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


"""
注意：本專案 UI 主流程以 `/multi-seal-test` 為唯一入口，僅保留 images 與多印鑑/PDF 任務所需 schema。
"""

