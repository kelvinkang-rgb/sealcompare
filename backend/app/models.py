"""
SQLAlchemy 資料庫模型
"""

from sqlalchemy import Column, String, Float, Boolean, DateTime, ForeignKey, JSON, Enum as SQLEnum, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
import enum

from app.database import Base


class ComparisonStatus(str, enum.Enum):
    """比對狀態枚舉"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Image(Base):
    """圖像模型"""
    __tablename__ = "images"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False, unique=True)
    file_size = Column(String(20), nullable=True)
    mime_type = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 印鑑檢測相關
    seal_detected = Column(Boolean, default=False, nullable=False)
    seal_confidence = Column(Float, nullable=True)
    seal_bbox = Column(JSON, nullable=True)  # {"x": int, "y": int, "width": int, "height": int}
    seal_center = Column(JSON, nullable=True)  # {"center_x": int, "center_y": int, "radius": float}
    
    # 多印鑑檢測相關（測試功能）
    multiple_seals = Column(JSON, nullable=True)  # [{"bbox": {...}, "center": {...}, "confidence": float}, ...]

    # PDF 相關（B 模式：PDF 任務 + 分頁）
    # - PDF 本體：is_pdf=True, pdf_page_count 有值
    # - PDF 分頁圖：source_pdf_id 指向 PDF Image，page_index 表示頁序（1-based）
    is_pdf = Column(Boolean, default=False, nullable=False)
    pdf_page_count = Column(Integer, nullable=True)
    source_pdf_id = Column(UUID(as_uuid=True), ForeignKey("images.id"), nullable=True)
    page_index = Column(Integer, nullable=True)
    
    # 自關聯：PDF 與 pages（避免前端/Schema recursion，這裡只做關聯方便查詢）
    source_pdf = relationship("Image", remote_side=[id], foreign_keys=[source_pdf_id], backref="pdf_pages")


class MultiSealComparisonTask(Base):
    """多印鑑比對任務模型"""
    __tablename__ = "multi_seal_comparison_tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_uid = Column(String(36), unique=True, nullable=False, index=True)  # 任務 UID（用於追蹤）
    image1_id = Column(UUID(as_uuid=True), ForeignKey("images.id"), nullable=False)
    
    status = Column(SQLEnum(ComparisonStatus), default=ComparisonStatus.PENDING, nullable=False)
    progress = Column(Float, nullable=True)  # 進度百分比 (0-100)
    progress_message = Column(String(500), nullable=True)  # 當前進度消息
    
    # 比對參數
    seal_image_ids = Column(JSON, nullable=False)  # 印鑑圖像 ID 列表
    threshold = Column(Float, default=0.83, nullable=False)
    similarity_ssim_weight = Column(Float, default=0.5, nullable=False)
    similarity_template_weight = Column(Float, default=0.35, nullable=False)
    pixel_similarity_weight = Column(Float, default=0.1, nullable=False)
    histogram_similarity_weight = Column(Float, default=0.05, nullable=False)
    
    # 比對結果
    results = Column(JSON, nullable=True)  # 比對結果列表
    total_count = Column(Integer, nullable=True)
    success_count = Column(Integer, nullable=True)
    
    # 錯誤信息
    error = Column(String(1000), nullable=True)
    error_trace = Column(String(5000), nullable=True)
    
    # 時間戳
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # 關係
    image1 = relationship("Image", foreign_keys=[image1_id])
