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


class VisualizationType(str, enum.Enum):
    """視覺化類型枚舉"""
    COMPARISON_IMAGE = "comparison_image"
    HEATMAP = "heatmap"
    OVERLAY1 = "overlay1"
    OVERLAY2 = "overlay2"


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
    
    # 關係
    comparisons_as_image1 = relationship("Comparison", foreign_keys="Comparison.image1_id", back_populates="image1")
    comparisons_as_image2 = relationship("Comparison", foreign_keys="Comparison.image2_id", back_populates="image2")


class Comparison(Base):
    """比對記錄模型"""
    __tablename__ = "comparisons"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image1_id = Column(UUID(as_uuid=True), ForeignKey("images.id"), nullable=False)
    image2_id = Column(UUID(as_uuid=True), ForeignKey("images.id"), nullable=False)
    image2_corrected_path = Column(String(500), nullable=True)
    
    status = Column(SQLEnum(ComparisonStatus), default=ComparisonStatus.PENDING, nullable=False)
    is_match = Column(Boolean, nullable=True)
    similarity = Column(Float, nullable=True)
    threshold = Column(Float, default=0.83, nullable=False)
    
    # 校正相關
    rotation_angle = Column(Float, nullable=True)
    similarity_before_correction = Column(Float, nullable=True)
    improvement = Column(Float, nullable=True)
    
    # 詳細指標（JSON 格式）
    details = Column(JSON, nullable=True)
    
    # 對齊指標
    center_offset = Column(Float, nullable=True)
    size_ratio = Column(Float, nullable=True)
    
    # 進度相關
    progress = Column(Float, nullable=True)  # 進度百分比 (0-100)
    progress_message = Column(String(500), nullable=True)  # 當前進度消息
    current_step = Column(String(100), nullable=True)  # 當前處理步驟
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    deleted_at = Column(DateTime, nullable=True)  # 軟刪除時間戳
    
    # 備註
    notes = Column(String(500), nullable=True)  # 備註或標籤
    
    # 關係
    image1 = relationship("Image", foreign_keys=[image1_id], back_populates="comparisons_as_image1")
    image2 = relationship("Image", foreign_keys=[image2_id], back_populates="comparisons_as_image2")
    visualizations = relationship("ComparisonVisualization", back_populates="comparison", cascade="all, delete-orphan")


class ComparisonVisualization(Base):
    """比對視覺化模型"""
    __tablename__ = "comparison_visualizations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    comparison_id = Column(UUID(as_uuid=True), ForeignKey("comparisons.id"), nullable=False)
    type = Column(SQLEnum(VisualizationType), nullable=False)
    file_path = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 關係
    comparison = relationship("Comparison", back_populates="visualizations")


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
