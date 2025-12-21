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
    
    def process_comparison(self, comparison_id: UUID, enable_rotation_search: bool = True, 
                          enable_translation_search: bool = True) -> Comparison:
        """
        處理比對（執行實際的圖像比對）
        
        Args:
            comparison_id: 比對 ID
            enable_rotation_search: 是否啟用旋轉搜索
            enable_translation_search: 是否啟用平移搜索
            
        Returns:
            Comparison 模型實例
        """
        db_comparison = self.db.query(Comparison).filter(Comparison.id == comparison_id).first()
        if not db_comparison:
            raise ValueError("比對記錄不存在")
        
        # 更新狀態為處理中
        db_comparison.status = ComparisonStatus.PROCESSING
        # 初始化處理階段信息
        # 注意：初始化階段在設置階段信息時就已經完成，所以直接標記為完成
        processing_stages = {
            'current_stage': 'loading_images',
            'stages': [
                {'name': 'initializing', 'label': '初始化', 'status': 'completed', 'progress': 100},
                {'name': 'loading_images', 'label': '載入圖像', 'status': 'in_progress', 'progress': 10},
                {'name': 'preprocessing', 'label': '圖像預處理', 'status': 'pending', 'progress': 0},
            ]
        }
        
        # 根據啟用的功能添加階段
        if enable_rotation_search:
            processing_stages['stages'].append({
                'name': 'rotation_search', 
                'label': '旋轉角度搜索', 
                'status': 'pending', 
                'progress': 0
            })
        if enable_translation_search:
            processing_stages['stages'].append({
                'name': 'translation_search', 
                'label': '平移位置搜索', 
                'status': 'pending', 
                'progress': 0
            })
        
        processing_stages['stages'].extend([
            {'name': 'similarity_calculation', 'label': '計算相似度', 'status': 'pending', 'progress': 0},
            {'name': 'saving_corrected', 'label': '保存校正後圖像', 'status': 'pending', 'progress': 0},
            {'name': 'alignment_metrics', 'label': '計算對齊指標', 'status': 'pending', 'progress': 0},
            {'name': 'generating_visualizations', 'label': '生成視覺化圖表', 'status': 'pending', 'progress': 0},
            {'name': 'completed', 'label': '完成', 'status': 'pending', 'progress': 0},
        ])
        
        # 更新 details 並提交
        if db_comparison.details is None:
            db_comparison.details = {}
        db_comparison.details['processing_stages'] = processing_stages
        self.db.commit()
        
        try:
            # 階段1: 載入圖像
            # 進度 20%: 獲取圖像路徑
            processing_stages['stages'][1]['progress'] = 20
            db_comparison.details['processing_stages'] = processing_stages
            self.db.commit()
            
            image1_path = db_comparison.image1.file_path
            image2_path = db_comparison.image2.file_path
            
            # 標準化路徑處理（確保路徑正確解析）
            image1_path_obj = Path(image1_path)
            image2_path_obj = Path(image2_path)
            
            # 進度 40%: 驗證圖像文件是否存在
            processing_stages['stages'][1]['progress'] = 40
            db_comparison.details['processing_stages'] = processing_stages
            self.db.commit()
            
            # 驗證圖像1文件是否存在
            if not image1_path or not image1_path_obj.exists():
                error_msg = f"圖像1文件不存在: {image1_path}"
                db_comparison.status = ComparisonStatus.FAILED
                db_comparison.details = db_comparison.details or {}
                db_comparison.details['error'] = error_msg
                processing_stages['current_stage'] = 'failed'
                # 保持當前進度（40%），只更新狀態為 failed
                for stage in processing_stages['stages']:
                    if stage['status'] == 'in_progress':
                        stage['status'] = 'failed'
                        # 不重置進度為0，保持當前進度
                db_comparison.details['processing_stages'] = processing_stages
                self.db.commit()
                raise ValueError(error_msg)
            
            # 驗證圖像2文件是否存在
            if not image2_path or not image2_path_obj.exists():
                error_msg = f"圖像2文件不存在: {image2_path}"
                db_comparison.status = ComparisonStatus.FAILED
                db_comparison.details = db_comparison.details or {}
                db_comparison.details['error'] = error_msg
                processing_stages['current_stage'] = 'failed'
                # 保持當前進度（40%），只更新狀態為 failed
                for stage in processing_stages['stages']:
                    if stage['status'] == 'in_progress':
                        stage['status'] = 'failed'
                        # 不重置進度為0，保持當前進度
                db_comparison.details['processing_stages'] = processing_stages
                self.db.commit()
                raise ValueError(error_msg)
            
            # 進度 60%: 讀取裁切區域信息（如果沒有 bbox，將使用整個圖像進行比對）
            processing_stages['stages'][1]['progress'] = 60
            db_comparison.details['processing_stages'] = processing_stages
            self.db.commit()
            
            bbox1 = db_comparison.image1.seal_bbox if db_comparison.image1.seal_bbox else None
            bbox2 = db_comparison.image2.seal_bbox if db_comparison.image2.seal_bbox else None
            
            # 進度 80%: 驗證文件可讀性（只檢查文件格式，不實際載入圖像數據）
            # 這樣可以避免重複載入，實際載入會在 compare_files 中進行
            processing_stages['stages'][1]['progress'] = 80
            db_comparison.details['processing_stages'] = processing_stages
            self.db.commit()
            
            # 簡單驗證文件是否為有效的圖像文件（通過檢查文件擴展名和基本讀取測試）
            try:
                # 只檢查文件是否可以打開（快速檢查），不載入完整圖像數據
                with open(image1_path_obj, 'rb') as f:
                    header = f.read(10)
                    if len(header) < 10:
                        raise ValueError(f"圖像1文件損壞或格式不正確: {image1_path}")
                
                with open(image2_path_obj, 'rb') as f:
                    header = f.read(10)
                    if len(header) < 10:
                        raise ValueError(f"圖像2文件損壞或格式不正確: {image2_path}")
            except (IOError, OSError) as e:
                error_msg = f"無法讀取圖像文件: {str(e)}"
                db_comparison.status = ComparisonStatus.FAILED
                db_comparison.details = db_comparison.details or {}
                db_comparison.details['error'] = error_msg
                processing_stages['current_stage'] = 'failed'
                # 保持當前進度（80%），只更新狀態為 failed
                for stage in processing_stages['stages']:
                    if stage['status'] == 'in_progress':
                        stage['status'] = 'failed'
                        # 不重置進度為0，保持當前進度
                db_comparison.details['processing_stages'] = processing_stages
                self.db.commit()
                raise ValueError(error_msg)
            
            # 進度 90%: 載入並裁切圖像（如果有 bbox，裁切後保存作為待比對圖像）
            processing_stages['stages'][1]['progress'] = 90
            db_comparison.details['processing_stages'] = processing_stages
            self.db.commit()
            
            # 載入原始圖像並進行裁切（如果有的話）
            img1_original = cv2.imread(str(image1_path_obj))
            img2_original = cv2.imread(str(image2_path_obj))
            
            if img1_original is None:
                raise ValueError(f"無法載入圖像1: {image1_path}")
            if img2_original is None:
                raise ValueError(f"無法載入圖像2: {image2_path}")
            
            # 裁切圖像並保存（全部使用裁切後的圖像進行後續處理）
            # 如果有 bbox，進行裁切；如果沒有 bbox，保存整個圖像作為"裁切後的圖像"
            corrected_dir = self.logs_dir / "corrected_images"
            corrected_dir.mkdir(parents=True, exist_ok=True)
            
            # 處理圖像1：裁切或保存整個圖像
            if bbox1:
                x1, y1, w1, h1 = bbox1['x'], bbox1['y'], bbox1['width'], bbox1['height']
                h_img1, w_img1 = img1_original.shape[:2]
                # 確保裁切區域在圖像範圍內
                x1 = max(0, min(x1, w_img1 - 1))
                y1 = max(0, min(y1, h_img1 - 1))
                w1 = min(w1, w_img1 - x1)
                h1 = min(h1, h_img1 - y1)
                if w1 > 0 and h1 > 0:
                    img1_cropped = img1_original[y1:y1+h1, x1:x1+w1]
                else:
                    # 如果 bbox 無效，使用整個圖像
                    img1_cropped = img1_original.copy()
            else:
                # 沒有 bbox，使用整個圖像
                img1_cropped = img1_original.copy()
            
            # 對圖像1進行去背景和對齊
            img1_cropped = self._remove_background_and_align(img1_cropped)
            
            # 保存裁切並對齊後的圖像1
            image1_cropped_path = corrected_dir / f"image1_cropped_{comparison_id}.jpg"
            cv2.imwrite(str(image1_cropped_path), img1_cropped)
            image1_for_comparison = str(image1_cropped_path)
            
            # 處理圖像2：裁切或保存整個圖像
            if bbox2:
                x2, y2, w2, h2 = bbox2['x'], bbox2['y'], bbox2['width'], bbox2['height']
                h_img2, w_img2 = img2_original.shape[:2]
                # 確保裁切區域在圖像範圍內
                x2 = max(0, min(x2, w_img2 - 1))
                y2 = max(0, min(y2, h_img2 - 1))
                w2 = min(w2, w_img2 - x2)
                h2 = min(h2, h_img2 - y2)
                if w2 > 0 and h2 > 0:
                    img2_cropped = img2_original[y2:y2+h2, x2:x2+w2]
                else:
                    # 如果 bbox 無效，使用整個圖像
                    img2_cropped = img2_original.copy()
            else:
                # 沒有 bbox，使用整個圖像
                img2_cropped = img2_original.copy()
            
            # 對圖像2進行去背景和對齊
            img2_cropped = self._remove_background_and_align(img2_cropped)
            
            # 保存裁切並對齊後的圖像2
            image2_cropped_path = corrected_dir / f"image2_cropped_{comparison_id}.jpg"
            cv2.imwrite(str(image2_cropped_path), img2_cropped)
            image2_for_comparison = str(image2_cropped_path)
            
            # bbox 設為 None，因為圖像已經裁切好了
            bbox1 = None
            bbox2 = None
            
            # 進度 100%: 載入圖像階段完成
            processing_stages['stages'][1]['status'] = 'completed'
            processing_stages['stages'][1]['progress'] = 100
            db_comparison.details['processing_stages'] = processing_stages
            self.db.commit()
            
            # 階段2: 圖像預處理（已完成，因為圖像已經在載入階段完成去背景和對齊）
            processing_stages['current_stage'] = 'preprocessing'
            processing_stages['stages'][2]['status'] = 'completed'
            processing_stages['stages'][2]['progress'] = 100
            db_comparison.details['processing_stages'] = processing_stages
            self.db.commit()
            
            # 執行比對（使用裁切並對齊後的圖像路徑，bbox 設為 None）
            comparator = SealComparator(threshold=db_comparison.threshold)
            
            # 標記旋轉搜索和平移搜索階段為完成（因為圖像已經對齊，不需要這些搜索）
            stage_index = 3
            if enable_rotation_search:
                for idx, stage in enumerate(processing_stages['stages']):
                    if stage['name'] == 'rotation_search':
                        processing_stages['stages'][idx]['status'] = 'completed'
                        processing_stages['stages'][idx]['progress'] = 100
                        break
                stage_index += 1
            
            if enable_translation_search:
                for idx, stage in enumerate(processing_stages['stages']):
                    if stage['name'] == 'translation_search':
                        processing_stages['stages'][idx]['status'] = 'completed'
                        processing_stages['stages'][idx]['progress'] = 100
                        break
                stage_index += 1
            
            db_comparison.details['processing_stages'] = processing_stages
            self.db.commit()
            
            # 使用裁切並對齊後的圖像進行比對（不再需要旋轉和平移搜索）
            is_match, similarity, details, img2_corrected, img1_corrected = comparator.compare_files(
                image1_for_comparison,
                image2_for_comparison,
                enable_rotation_search=False,  # 圖像已對齊，不需要旋轉搜索
                enable_translation_search=False,  # 圖像已對齊，不需要平移搜索
                bbox1=None,  # 已經裁切並對齊好了，不需要再傳入 bbox
                bbox2=None   # 已經裁切並對齊好了，不需要再傳入 bbox
            )
            
            # 階段: 計算相似度
            processing_stages['current_stage'] = 'similarity_calculation'
            for idx, stage in enumerate(processing_stages['stages']):
                if stage['name'] == 'similarity_calculation':
                    processing_stages['stages'][idx]['status'] = 'in_progress'
                    processing_stages['stages'][idx]['progress'] = 50
                    db_comparison.details['processing_stages'] = processing_stages
                    self.db.commit()
                    break
            
            # 完成相似度計算
            for idx, stage in enumerate(processing_stages['stages']):
                if stage['name'] == 'similarity_calculation':
                    processing_stages['stages'][idx]['status'] = 'completed'
                    processing_stages['stages'][idx]['progress'] = 100
                    break
            
            # 保存校正後的圖像1和圖像2（裁切後的圖像已經在載入階段保存了）
            processing_stages['current_stage'] = 'saving_corrected'
            for idx, stage in enumerate(processing_stages['stages']):
                if stage['name'] == 'saving_corrected':
                    processing_stages['stages'][idx]['status'] = 'in_progress'
                    processing_stages['stages'][idx]['progress'] = 50
                    db_comparison.details['processing_stages'] = processing_stages
                    self.db.commit()
                    break
            
            # 裁切後的圖像路徑已在載入階段保存（image1_cropped_path 和 image2_cropped_path 已經是 Path 對象或 None）
            
            # 保存校正後的圖像1和圖像2
            image1_corrected_path = None
            image2_corrected_path = None
            corrected_dir = self.logs_dir / "corrected_images"
            corrected_dir.mkdir(parents=True, exist_ok=True)
            
            if img1_corrected is not None:
                image1_corrected_path = corrected_dir / f"image1_corrected_{comparison_id}.jpg"
                cv2.imwrite(str(image1_corrected_path), img1_corrected)
            
            if img2_corrected is not None:
                image2_corrected_path = corrected_dir / f"image2_corrected_{comparison_id}.jpg"
                cv2.imwrite(str(image2_corrected_path), img2_corrected)
                db_comparison.image2_corrected_path = str(image2_corrected_path)
            
            for idx, stage in enumerate(processing_stages['stages']):
                if stage['name'] == 'saving_corrected':
                    processing_stages['stages'][idx]['status'] = 'completed'
                    processing_stages['stages'][idx]['progress'] = 100
                    break
            
            # 更新比對結果
            db_comparison.is_match = is_match
            db_comparison.similarity = similarity
            db_comparison.rotation_angle = details.get('rotation_angle')
            # 存儲平移偏移量（如果存在）
            translation_offset = details.get('translation_offset')
            if translation_offset:
                # 將 translation_offset 存儲在 details 中，同時也可以單獨存儲
                pass  # 目前存儲在 details 中，如果需要可以添加到模型欄位
            db_comparison.similarity_before_correction = details.get('similarity_before_correction')
            db_comparison.improvement = details.get('improvement')
            details['processing_stages'] = processing_stages
            db_comparison.details = details
            
            # 階段: 計算對齊指標
            processing_stages['current_stage'] = 'alignment_metrics'
            for idx, stage in enumerate(processing_stages['stages']):
                if stage['name'] == 'alignment_metrics':
                    processing_stages['stages'][idx]['status'] = 'in_progress'
                    processing_stages['stages'][idx]['progress'] = 50
                    db_comparison.details['processing_stages'] = processing_stages
                    self.db.commit()
                    break
            
            # 使用校正後的圖像計算對齊指標
            # 優先使用校正後的圖像，否則使用裁切後的圖像（所有處理都使用裁切後的圖像）
            if image1_corrected_path and Path(image1_corrected_path).exists():
                img1_corr = cv2.imread(str(image1_corrected_path))
            else:
                # 使用裁切後的圖像1
                img1_corr = cv2.imread(str(image1_cropped_path))
            
            # 使用裁切後的圖像2作為原始圖像2
            img2_orig = cv2.imread(str(image2_cropped_path))
            
            # 使用校正後的圖像2（如果存在）
            if image2_corrected_path and Path(image2_corrected_path).exists():
                img2_corr = cv2.imread(str(image2_corrected_path))
            else:
                img2_corr = None
            
            alignment_metrics = calculate_alignment_metrics(
                img1_corr, img2_orig, img2_corr, 
                details.get('rotation_angle'),
                details.get('translation_offset')
            )
            db_comparison.center_offset = alignment_metrics.get('center_offset', 0.0)
            db_comparison.size_ratio = details.get('size_ratio', 1.0)
            
            for idx, stage in enumerate(processing_stages['stages']):
                if stage['name'] == 'alignment_metrics':
                    processing_stages['stages'][idx]['status'] = 'completed'
                    processing_stages['stages'][idx]['progress'] = 100
                    break
            
            # 階段: 生成視覺化
            processing_stages['current_stage'] = 'generating_visualizations'
            for idx, stage in enumerate(processing_stages['stages']):
                if stage['name'] == 'generating_visualizations':
                    processing_stages['stages'][idx]['status'] = 'in_progress'
                    processing_stages['stages'][idx]['progress'] = 50
                    db_comparison.details['processing_stages'] = processing_stages
                    self.db.commit()
                    break
            
            # 使用裁切後的圖像路徑生成視覺化（所有處理都使用裁切後的圖像）
            # image1_path_for_viz: 優先使用校正後的圖像1，否則使用裁切後的圖像1
            if image1_corrected_path and Path(image1_corrected_path).exists():
                image1_path_for_viz = str(image1_corrected_path)
            else:
                image1_path_for_viz = str(image1_cropped_path)
            
            # image2_path_for_viz: 使用裁切後的圖像2作為原始圖像2（用於並排對比顯示）
            image2_path_for_viz = str(image2_cropped_path)
            
            self._generate_visualizations(
                db_comparison,
                image1_path_for_viz,
                image2_path_for_viz,
                str(image2_corrected_path) if image2_corrected_path and Path(image2_corrected_path).exists() else None,
                details.get('rotation_angle')
            )
            
            for idx, stage in enumerate(processing_stages['stages']):
                if stage['name'] == 'generating_visualizations':
                    processing_stages['stages'][idx]['status'] = 'completed'
                    processing_stages['stages'][idx]['progress'] = 100
                    break
            
            # 完成
            processing_stages['current_stage'] = 'completed'
            for idx, stage in enumerate(processing_stages['stages']):
                if stage['name'] == 'completed':
                    processing_stages['stages'][idx]['status'] = 'completed'
                    processing_stages['stages'][idx]['progress'] = 100
                    break
            db_comparison.details['processing_stages'] = processing_stages
            db_comparison.status = ComparisonStatus.COMPLETED
            
            self.db.commit()
            self.db.refresh(db_comparison)
            
            return db_comparison
            
        except Exception as e:
            # 記錄詳細錯誤信息
            import traceback
            error_trace = traceback.format_exc()
            error_msg = f"比對處理失敗: {str(e)}"
            print(f"錯誤詳情: {error_msg}")
            print(f"錯誤堆疊: {error_trace}")
            
            # 更新階段狀態為失敗
            if 'processing_stages' in (db_comparison.details or {}):
                processing_stages = db_comparison.details['processing_stages']
                for idx, stage in enumerate(processing_stages.get('stages', [])):
                    if stage['status'] == 'in_progress':
                        processing_stages['stages'][idx]['status'] = 'failed'
                        # 保持當前進度，不重置為0
                        processing_stages['current_stage'] = 'failed'
                        break
                db_comparison.details['processing_stages'] = processing_stages
            
            # 保存錯誤信息到 details
            if db_comparison.details is None:
                db_comparison.details = {}
            db_comparison.details['error'] = error_msg
            db_comparison.details['error_trace'] = error_trace
            
            db_comparison.status = ComparisonStatus.FAILED
            self.db.commit()
            raise e
    
    def _remove_background_and_align(self, image: np.ndarray) -> np.ndarray:
        """
        去背景並對齊圖像
        
        Args:
            image: 裁切後的圖像（numpy array）
            
        Returns:
            去背景並對齊後的圖像
        """
        comparator = SealComparator()
        
        # 去背景
        try:
            image = comparator._auto_detect_bounds_and_remove_background(image)
        except Exception as e:
            print(f"警告：自動外框偵測失敗，使用原圖: {str(e)}")
            # 如果處理失敗，使用原圖繼續處理
            pass
        
        # 對齊（使用 _align_image1 的邏輯，因為兩個圖像都需要平移+旋轉）
        try:
            image_aligned, _, _ = comparator._align_image1(image, bbox=None)
            return image_aligned
        except Exception as e:
            print(f"警告：圖像對齊失敗，使用原圖: {str(e)}")
            # 如果對齊失敗，返回去背景後的圖像
            return image
    
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
        # 從 details 中獲取平移信息
        translation_offset = None
        if comparison.details and isinstance(comparison.details, dict):
            translation_offset = comparison.details.get('translation_offset')
        
        comparison_url = create_correction_comparison(
            image1_path,
            image2_path,
            image2_corrected_path,
            comparison_path,
            record_id,
            rotation_angle,
            translation_offset
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

