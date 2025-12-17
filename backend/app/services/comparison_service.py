"""
比對服務
"""

from pathlib import Path
from sqlalchemy.orm import Session
from typing import Optional
from uuid import UUID
import cv2
import numpy as np

from app.models import Comparison, Image, ComparisonVisualization, ComparisonStatus, VisualizationType
from app.schemas import ComparisonCreate, ComparisonResponse
from app.config import settings
import sys
from pathlib import Path
# 添加 core 目錄到路徑
core_path = Path(__file__).parent.parent.parent / "core"
sys.path.insert(0, str(core_path))

from seal_compare import SealComparator
from verification import (
    create_correction_comparison,
    create_difference_heatmap,
    calculate_alignment_metrics
)
from overlay import create_overlay_image
from overlay import create_overlay_image


class ComparisonService:
    """比對服務類"""
    
    def __init__(self, db: Session):
        self.db = db
        self.logs_dir = Path(settings.LOGS_DIR)
        self.upload_dir = Path(settings.UPLOAD_DIR)
    
    def create_comparison(self, comparison_data: ComparisonCreate) -> Comparison:
        """
        創建比對記錄
        
        Args:
            comparison_data: 比對創建數據
            
        Returns:
            Comparison 模型實例
        """
        # 驗證圖像存在
        image1 = self.db.query(Image).filter(Image.id == comparison_data.image1_id).first()
        image2 = self.db.query(Image).filter(Image.id == comparison_data.image2_id).first()
        
        if not image1 or not image2:
            raise ValueError("圖像不存在")
        
        # 創建比對記錄
        db_comparison = Comparison(
            image1_id=comparison_data.image1_id,
            image2_id=comparison_data.image2_id,
            threshold=comparison_data.threshold,
            status=ComparisonStatus.PENDING
        )
        
        self.db.add(db_comparison)
        self.db.commit()
        self.db.refresh(db_comparison)
        
        return db_comparison
    
    def process_comparison(self, comparison_id: UUID, enable_rotation_search: bool = True) -> Comparison:
        """
        處理比對（執行實際的圖像比對）
        
        Args:
            comparison_id: 比對 ID
            enable_rotation_search: 是否啟用旋轉搜索
            
        Returns:
            Comparison 模型實例
        """
        db_comparison = self.db.query(Comparison).filter(Comparison.id == comparison_id).first()
        if not db_comparison:
            raise ValueError("比對記錄不存在")
        
        # 更新狀態為處理中
        db_comparison.status = ComparisonStatus.PROCESSING
        self.db.commit()
        
        try:
            # 獲取圖像路徑
            image1_path = db_comparison.image1.file_path
            image2_path = db_comparison.image2.file_path
            
            # 執行比對
            comparator = SealComparator(threshold=db_comparison.threshold)
            is_match, similarity, details, img2_corrected = comparator.compare_files(
                image1_path,
                image2_path,
                enable_rotation_search=enable_rotation_search
            )
            
            # 保存校正後的圖像2（如果存在）
            image2_corrected_path = None
            if img2_corrected is not None:
                corrected_dir = self.logs_dir / "corrected_images"
                corrected_dir.mkdir(parents=True, exist_ok=True)
                image2_corrected_path = corrected_dir / f"image2_corrected_{comparison_id}.jpg"
                cv2.imwrite(str(image2_corrected_path), img2_corrected)
                db_comparison.image2_corrected_path = str(image2_corrected_path)
            
            # 更新比對結果
            db_comparison.is_match = is_match
            db_comparison.similarity = similarity
            db_comparison.rotation_angle = details.get('rotation_angle')
            db_comparison.similarity_before_correction = details.get('similarity_before_correction')
            db_comparison.improvement = details.get('improvement')
            db_comparison.details = details
            db_comparison.status = ComparisonStatus.COMPLETED
            
            # 計算對齊指標
            img1 = cv2.imread(image1_path)
            img2_orig = cv2.imread(image2_path)
            img2_corr = cv2.imread(str(image2_corrected_path)) if image2_corrected_path else None
            
            alignment_metrics = calculate_alignment_metrics(
                img1, img2_orig, img2_corr, details.get('rotation_angle')
            )
            db_comparison.center_offset = alignment_metrics.get('center_offset', 0.0)
            db_comparison.size_ratio = details.get('size_ratio', 1.0)
            
            # 生成視覺化
            self._generate_visualizations(
                db_comparison,
                image1_path,
                image2_path,
                str(image2_corrected_path) if image2_corrected_path else None,
                details.get('rotation_angle')
            )
            
            self.db.commit()
            self.db.refresh(db_comparison)
            
            return db_comparison
            
        except Exception as e:
            db_comparison.status = ComparisonStatus.FAILED
            self.db.commit()
            raise e
    
    def _generate_visualizations(
        self,
        comparison: Comparison,
        image1_path: str,
        image2_path: str,
        image2_corrected_path: Optional[str],
        rotation_angle: Optional[float]
    ):
        """
        生成視覺化圖像
        
        Args:
            comparison: 比對記錄
            image1_path: 圖像1路徑
            image2_path: 圖像2原始路徑
            image2_corrected_path: 圖像2校正後路徑
            rotation_angle: 旋轉角度
        """
        record_id = str(comparison.id).replace('-', '')[:8]  # 使用 ID 的前8位作為記錄 ID
        
        # 生成並排對比圖
        comparison_path = self.logs_dir / "comparisons"
        comparison_url = create_correction_comparison(
            image1_path,
            image2_path,
            image2_corrected_path,
            comparison_path,
            record_id,
            rotation_angle
        )
        
        if comparison_url:
            vis = ComparisonVisualization(
                comparison_id=comparison.id,
                type=VisualizationType.COMPARISON_IMAGE,
                file_path=str(comparison_path / Path(comparison_url).name)
            )
            self.db.add(vis)
        
        # 生成差異熱力圖
        heatmap_path = self.logs_dir / "heatmaps"
        heatmap_url, heatmap_stats = create_difference_heatmap(
            image1_path,
            image2_corrected_path,
            image2_path,
            heatmap_path,
            record_id
        )
        
        if heatmap_url:
            vis = ComparisonVisualization(
                comparison_id=comparison.id,
                type=VisualizationType.HEATMAP,
                file_path=str(heatmap_path / Path(heatmap_url).name)
            )
            self.db.add(vis)
        
        # 生成疊圖
        overlay_path = self.logs_dir / "overlays"
        overlay1_path, overlay2_path = create_overlay_image(
            image1_path,
            image2_path,
            overlay_path,
            record_id,
            image2_corrected_path
        )
        
        if overlay1_path:
            vis = ComparisonVisualization(
                comparison_id=comparison.id,
                type=VisualizationType.OVERLAY1,
                file_path=overlay1_path
            )
            self.db.add(vis)
        
        if overlay2_path:
            vis = ComparisonVisualization(
                comparison_id=comparison.id,
                type=VisualizationType.OVERLAY2,
                file_path=overlay2_path
            )
            self.db.add(vis)
    
    def get_comparison(self, comparison_id: UUID, include_deleted: bool = False) -> Optional[Comparison]:
        """
        獲取比對記錄
        
        Args:
            comparison_id: 比對 ID
            include_deleted: 是否包含已刪除的記錄
            
        Returns:
            Comparison 模型實例或 None
        """
        query = self.db.query(Comparison).filter(Comparison.id == comparison_id)
        if not include_deleted:
            query = query.filter(Comparison.deleted_at.is_(None))
        return query.first()
    
    def list_comparisons(self, skip: int = 0, limit: int = 100, include_deleted: bool = False):
        """
        列出比對記錄
        
        Args:
            skip: 跳過數量
            limit: 限制數量
            include_deleted: 是否包含已刪除的記錄
            
        Returns:
            比對記錄列表
        """
        query = self.db.query(Comparison)
        if not include_deleted:
            query = query.filter(Comparison.deleted_at.is_(None))
        return query.order_by(Comparison.created_at.desc()).offset(skip).limit(limit).all()
    
    def update_comparison(self, comparison_id: UUID, update_data: dict) -> Comparison:
        """
        更新比對記錄
        
        Args:
            comparison_id: 比對 ID
            update_data: 更新數據字典
            
        Returns:
            更新後的 Comparison 模型實例
        """
        comparison = self.get_comparison(comparison_id)
        if not comparison:
            raise ValueError("比對記錄不存在或已刪除")
        
        # 更新允許修改的欄位
        if 'threshold' in update_data and update_data['threshold'] is not None:
            comparison.threshold = update_data['threshold']
            # 如果比對已完成，重新判斷匹配狀態
            if comparison.status == ComparisonStatus.COMPLETED and comparison.similarity is not None:
                comparison.is_match = comparison.similarity >= comparison.threshold
        
        if 'notes' in update_data:
            comparison.notes = update_data.get('notes')
        
        if 'status' in update_data and update_data['status'] is not None:
            comparison.status = update_data['status']
        
        self.db.commit()
        self.db.refresh(comparison)
        
        return comparison
    
    def delete_comparison(self, comparison_id: UUID) -> bool:
        """
        刪除比對記錄（軟刪除）
        
        Args:
            comparison_id: 比對 ID
            
        Returns:
            是否成功刪除
        """
        from datetime import datetime
        
        comparison = self.get_comparison(comparison_id)
        if not comparison:
            return False
        
        # 軟刪除：設置 deleted_at 時間戳
        comparison.deleted_at = datetime.utcnow()
        self.db.commit()
        
        return True
    
    def restore_comparison(self, comparison_id: UUID) -> bool:
        """
        恢復已刪除的比對記錄
        
        Args:
            comparison_id: 比對 ID
            
        Returns:
            是否成功恢復
        """
        comparison = self.db.query(Comparison).filter(Comparison.id == comparison_id).first()
        if not comparison:
            return False
        
        if comparison.deleted_at is None:
            return False  # 記錄未被刪除
        
        comparison.deleted_at = None
        self.db.commit()
        
        return True

