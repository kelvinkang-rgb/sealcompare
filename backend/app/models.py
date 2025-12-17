"""
SQLAlchemy 資料庫模型
"""

from sqlalchemy import Column, String, Float, Boolean, DateTime, ForeignKey, JSON, Enum as SQLEnum
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
    threshold = Column(Float, default=0.95, nullable=False)
    
    # 校正相關
    rotation_angle = Column(Float, nullable=True)
    similarity_before_correction = Column(Float, nullable=True)
    improvement = Column(Float, nullable=True)
    
    # 詳細指標（JSON 格式）
    details = Column(JSON, nullable=True)
    
    # 對齊指標
    center_offset = Column(Float, nullable=True)
    size_ratio = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    deleted_at = Column(DateTime, nullable=True)  # 軟刪除時間戳
    
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

