"""
圖像服務
"""

from pathlib import Path
from sqlalchemy.orm import Session
from typing import Optional, Dict, List, Tuple, Callable
from uuid import UUID
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import logging
import traceback

from app.models import Image, MultiSealComparisonTask, ComparisonStatus
from app.schemas import ImageCreate, ImageResponse
from app.utils.image_utils import save_uploaded_file, get_file_size, delete_file
from app.utils.seal_detector import detect_seal_location, detect_multiple_seals, detect_seals_with_rotation_matching
import sys
from pathlib import Path as PathLib
core_path = PathLib(__file__).parent.parent.parent / "core"
sys.path.insert(0, str(core_path))
from seal_compare import SealComparator
from overlay import create_overlay_image, calculate_mask_statistics, calculate_mask_based_similarity
from verification import create_difference_heatmap
from fastapi import UploadFile
from app.config import settings
from app.exceptions import (
    ImageNotFoundError,
    ImageFileNotFoundError,
    InvalidBboxError,
    InvalidBboxSizeError,
    InvalidCenterError,
    InvalidSealDataError,
    ImageNotMarkedError,
    ImageReadError,
    CropAreaTooSmallError,
    MultiSealComparisonTaskNotFoundError
)
import numpy as np

# 配置日誌記錄器
logger = logging.getLogger(__name__)


class ImageService:
    """圖像服務類"""
    
    def __init__(self, db: Session):
        self.db = db
        self.upload_dir = Path(settings.UPLOAD_DIR)
    
    def create_image(self, upload_file: UploadFile) -> Image:
        """
        創建圖像記錄
        
        Args:
            upload_file: 上傳的文件
            
        Returns:
            Image 模型實例
        """
        # 保存文件
        file_path, filename = save_uploaded_file(upload_file, self.upload_dir)
        
        # 獲取文件信息
        file_size = get_file_size(file_path)
        
        # 檢測 MIME 類型
        mime_type = upload_file.content_type or "application/octet-stream"
        
        # 創建資料庫記錄
        db_image = Image(
            filename=upload_file.filename or filename,
            file_path=str(file_path),
            file_size=file_size,
            mime_type=mime_type
        )
        
        self.db.add(db_image)
        self.db.commit()
        self.db.refresh(db_image)
        
        return db_image
    
    def get_image(self, image_id: UUID) -> Optional[Image]:
        """
        獲取圖像
        
        Args:
            image_id: 圖像 ID
            
        Returns:
            Image 模型實例或 None
        """
        return self.db.query(Image).filter(Image.id == image_id).first()
    
    def delete_image(self, image_id: UUID) -> bool:
        """
        刪除圖像
        
        Args:
            image_id: 圖像 ID
            
        Returns:
            是否成功刪除
        """
        db_image = self.get_image(image_id)
        if not db_image:
            return False
        
        # 刪除文件
        file_path = Path(db_image.file_path)
        if file_path.exists():
            delete_file(file_path)
        
        # 刪除資料庫記錄
        self.db.delete(db_image)
        self.db.commit()
        
        return True
    
    def verify_image(self, image_path: str) -> bool:
        """
        驗證圖像是否有效
        
        Args:
            image_path: 圖像路徑
            
        Returns:
            是否有效
        """
        try:
            img = cv2.imread(image_path)
            return img is not None
        except Exception:
            return False
    
    def detect_seal(self, image_id: UUID) -> Dict:
        """
        檢測圖像中的印鑑位置
        
        Args:
            image_id: 圖像 ID
            
        Returns:
            檢測結果字典
        """
        db_image = self.get_image(image_id)
        if not db_image:
            raise ImageNotFoundError("圖像不存在")
        
        file_path = Path(db_image.file_path)
        if not file_path.exists():
            raise ImageFileNotFoundError("圖像文件不存在")
        
        # 執行檢測（帶超時保護）
        detection_result = detect_seal_location(str(file_path), timeout=3.0)
        
        # 如果檢測成功，更新資料庫（但不強制，允許用戶手動調整）
        if detection_result.get('detected'):
            db_image.seal_detected = True
            db_image.seal_confidence = detection_result.get('confidence')
            db_image.seal_bbox = detection_result.get('bbox')
            db_image.seal_center = detection_result.get('center')
            self.db.commit()
            self.db.refresh(db_image)
        
        return detection_result
    
    def update_seal_location(
        self, 
        image_id: UUID, 
        bbox: Optional[Dict] = None,
        center: Optional[Dict] = None,
        confidence: Optional[float] = None
    ) -> Image:
        """
        更新用戶確認的印鑑位置
        
        Args:
            image_id: 圖像 ID
            bbox: 邊界框 {"x": int, "y": int, "width": int, "height": int}
            center: 中心點 {"center_x": int, "center_y": int, "radius": float}
            confidence: 置信度（可選）
            
        Returns:
            更新後的 Image 模型實例
        """
        db_image = self.get_image(image_id)
        if not db_image:
            raise ImageNotFoundError("圖像不存在")
        
        # 驗證數據
        if bbox:
            if not all(k in bbox for k in ['x', 'y', 'width', 'height']):
                raise InvalidBboxError("邊界框格式錯誤")
            if bbox['width'] < 10 or bbox['height'] < 10:
                raise InvalidBboxSizeError("邊界框尺寸太小")
        
        if center:
            if not all(k in center for k in ['center_x', 'center_y', 'radius']):
                raise InvalidCenterError("中心點格式錯誤")
        
        # 更新資料
        if bbox:
            db_image.seal_bbox = bbox
        if center:
            db_image.seal_center = center
        if confidence is not None:
            db_image.seal_confidence = confidence
        
        db_image.seal_detected = True
        self.db.commit()
        self.db.refresh(db_image)
        
        return db_image
    
    def detect_multiple_seals(self, image_id: UUID, max_seals: int = 10) -> Dict:
        """
        檢測圖像中的多個印鑑位置
        
        Args:
            image_id: 圖像 ID
            max_seals: 最大檢測數量，默認10
            
        Returns:
            檢測結果字典
        """
        db_image = self.get_image(image_id)
        if not db_image:
            raise ImageNotFoundError("圖像不存在")
        
        file_path = Path(db_image.file_path)
        if not file_path.exists():
            raise ImageFileNotFoundError("圖像文件不存在")
        
        # 執行多印鑑檢測（帶超時保護）
        detection_result = detect_multiple_seals(str(file_path), timeout=5.0, max_seals=max_seals)
        
        return detection_result
    
    def save_multiple_seals(self, image_id: UUID, seals: List[Dict]) -> Image:
        """
        保存多個印鑑位置到資料庫
        
        Args:
            image_id: 圖像 ID
            seals: 印鑑列表，每個元素包含 bbox, center, confidence
            
        Returns:
            更新後的 Image 模型實例
        """
        db_image = self.get_image(image_id)
        if not db_image:
            raise ImageNotFoundError("圖像不存在")
        
        # 驗證數據格式
        normalized_seals = []
        for seal in seals:
            if 'bbox' not in seal or 'center' not in seal:
                raise InvalidSealDataError("印鑑數據格式錯誤")
            
            bbox = seal['bbox']
            center = seal['center']
            confidence = seal.get('confidence', 0.5)
            
            # 驗證邊界框
            if not all(k in bbox for k in ['x', 'y', 'width', 'height']):
                raise InvalidBboxError("邊界框格式錯誤")
            if bbox['width'] < 10 or bbox['height'] < 10:
                continue  # 跳過太小的框
            
            # 驗證中心點
            if not all(k in center for k in ['center_x', 'center_y', 'radius']):
                raise InvalidCenterError("中心點格式錯誤")
            
            normalized_seals.append({
                'bbox': bbox,
                'center': center,
                'confidence': float(confidence)
            })
        
        # 更新資料庫
        db_image.multiple_seals = normalized_seals
        self.db.commit()
        self.db.refresh(db_image)
        
        return db_image
    
    def crop_seals(self, image_id: UUID, seals: List[Dict], margin: int = 15) -> List[UUID]:
        """
        裁切圖像中的多個印鑑區域並保存為獨立圖像
        
        Args:
            image_id: 原圖像 ID
            seals: 印鑑列表，每個元素包含 bbox, center, confidence
            margin: 邊距（像素），默認15。邊距用於確保裁切後的圖像邊緣有足夠的背景區域，
                   供 remove_background_and_make_transparent 函數檢測背景色（至少需要5像素邊緣）
            
        Returns:
            裁切後的圖像 ID 列表
        """
        db_image = self.get_image(image_id)
        if not db_image:
            raise ImageNotFoundError("圖像不存在")
        
        file_path = Path(db_image.file_path)
        if not file_path.exists():
            raise ImageFileNotFoundError("圖像文件不存在")
        
        # 讀取原圖
        image = cv2.imread(str(file_path))
        if image is None:
            raise ImageReadError("無法讀取圖像文件")
        
        h, w = image.shape[:2]
        cropped_image_ids = []
        
        # 裁切每個印鑑
        for idx, seal in enumerate(seals):
            bbox = seal['bbox']
            
            # 計算裁切區域（添加邊距）
            x = max(0, bbox['x'] - margin)
            y = max(0, bbox['y'] - margin)
            crop_width = min(w - x, bbox['width'] + 2 * margin)
            crop_height = min(h - y, bbox['height'] + 2 * margin)
            
            # 確保不超出邊界
            if x + crop_width > w:
                crop_width = w - x
            if y + crop_height > h:
                crop_height = h - y
            
            if crop_width < 10 or crop_height < 10:
                continue  # 跳過太小的區域
            
            # 裁切圖像
            cropped = image[y:y+crop_height, x:x+crop_width]
            
            # 生成文件名
            original_name = Path(db_image.filename).stem
            extension = Path(db_image.filename).suffix or '.jpg'
            cropped_filename = f"{original_name}_seal_{idx+1}{extension}"
            
            # 保存裁切後的圖像
            cropped_file_path = self.upload_dir / cropped_filename
            
            # 確保文件名唯一
            counter = 1
            while cropped_file_path.exists():
                cropped_filename = f"{original_name}_seal_{idx+1}_{counter}{extension}"
                cropped_file_path = self.upload_dir / cropped_filename
                counter += 1
            
            # 保存圖像
            cv2.imwrite(str(cropped_file_path), cropped)
            
            # 創建新的圖像記錄
            cropped_image = Image(
                filename=cropped_filename,
                file_path=str(cropped_file_path),
                file_size=get_file_size(cropped_file_path),
                mime_type=db_image.mime_type
            )
            
            self.db.add(cropped_image)
            self.db.commit()
            self.db.refresh(cropped_image)
            
            cropped_image_ids.append(cropped_image.id)
        
        return cropped_image_ids
    
    def _remove_background_and_align(
        self, 
        image: np.ndarray, 
        reference_image: Optional[np.ndarray] = None,
        is_image2: bool = False,
        rotation_range: float = 15.0,
        translation_range: int = 100
    ) -> Tuple[np.ndarray, Optional[float], Optional[Tuple[int, int]], Optional[float], Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """
        去背景並對齊圖像（與 comparison_service 保持一致）
        
        Args:
            image: 待處理圖像（numpy array）
            reference_image: 參考圖像（用於圖像2的對齊優化，應該是已去背景的圖像）
            is_image2: 是否為圖像2（需要相對於圖像1優化）
            
        Returns:
            如果 is_image2=True 且 reference_image 不為 None:
                (對齊後的圖像, 旋轉角度, (x偏移, y偏移), 相似度, 詳細指標, 對齊階段時間)
            否則:
                (去背景後的圖像, None, None, None, None, None)
        """
        import time
        comparator = SealComparator()
        
        # 去背景
        remove_bg_start = time.time()
        remove_bg_timing: Dict[str, float] = {}
        try:
            # 需要完整去背景步驟時間（step1..step9 + remove_background_total）
            image, remove_bg_timing = comparator._auto_detect_bounds_and_remove_background(image, return_timing=True)
        except Exception as e:
            print(f"警告：自動外框偵測失敗，使用原圖: {str(e)}")
            # 如果處理失敗，使用原圖繼續處理
            remove_bg_timing = {}
        remove_bg_time = time.time() - remove_bg_start
        
        # 對齊處理
        if is_image2 and reference_image is not None:
            # 圖像2：相對於圖像1進行優化對齊
            try:
                # reference_image 已經是去背景後的圖像（img1_no_bg），不需要再次處理
                image_aligned, angle, offset, similarity, metrics, alignment_timing = comparator._align_image2_to_image1(
                    reference_image, image, rotation_range=rotation_range, translation_range=translation_range
                )
                if alignment_timing is None:
                    alignment_timing = {}
                # 合併去背景「詳細步驟」時間到 alignment_timing，供前端顯示
                # 注意：多印鑑頁面依賴 result.timing.alignment_stages 來呈現這些資訊
                alignment_timing.update(remove_bg_timing)
                # 向後兼容：保留 remove_background 作為總時間
                if 'remove_background_total' in remove_bg_timing and remove_bg_timing.get('remove_background_total', 0.0) > 0:
                    alignment_timing['remove_background'] = float(remove_bg_timing['remove_background_total'])
                else:
                    alignment_timing['remove_background'] = float(remove_bg_time)
                return image_aligned, angle, offset, similarity, metrics, alignment_timing
            except Exception as e:
                print(f"警告：圖像2對齊優化失敗，使用原圖: {str(e)}")
                # 如果對齊失敗，直接返回原圖
                alignment_timing = dict(remove_bg_timing) if remove_bg_timing else {}
                if 'remove_background_total' in alignment_timing and alignment_timing.get('remove_background_total', 0.0) > 0:
                    alignment_timing['remove_background'] = float(alignment_timing['remove_background_total'])
                else:
                    alignment_timing['remove_background'] = float(remove_bg_time)
                return image, None, None, None, None, alignment_timing
        else:
            # 圖像1：只去背景，不對齊
            # 回傳去背景 timing（即使目前呼叫端不使用，也保持資訊完整）
            return image, None, None, None, None, remove_bg_timing
    
    def compare_image1_with_seals(
        self, 
        image1_id: UUID, 
        seal_image_ids: List[UUID],
        threshold: float = 0.95,
        similarity_ssim_weight: float = 0.5,
        similarity_template_weight: float = 0.35,
        pixel_similarity_weight: float = 0.1,
        histogram_similarity_weight: float = 0.05,
        overlap_weight: float = 0.5,
        pixel_diff_penalty_weight: float = 0.3,
        unique_region_penalty_weight: float = 0.2,
        rotation_range: float = 15.0,
        translation_range: int = 100,
        task_uid: Optional[str] = None,
        task_update_callback: Optional[Callable[[Dict, int, int], None]] = None
    ) -> List[Dict]:
        """
        將圖像1與多個裁切的印鑑圖像進行比對
        
        注意：圖像1會根據 seal_bbox 自動裁切，然後與每個裁切的印鑑圖像進行比對。
        每個印鑑都會進行去背景和旋轉對齊處理。
        
        Args:
            image1_id: 圖像1 ID（必須已標記印鑑位置 seal_bbox）
            seal_image_ids: 裁切後的印鑑圖像 ID 列表（從圖像2中裁切出的多個印鑑）
            threshold: 相似度閾值
            
        Returns:
            比對結果列表，每個元素包含：
            - seal_index: 印鑑索引
            - seal_image_id: 印鑑圖像 ID
            - similarity: 相似度
            - is_match: 是否匹配
            - overlay1_path: 疊圖1路徑（使用對齊後的圖像）
            - overlay2_path: 疊圖2路徑（使用對齊後的圖像）
            - heatmap_path: 熱力圖路徑（使用對齊後的圖像）
            - alignment_success: 是否成功對齊
            - alignment_angle: 旋轉角度（度）
            - alignment_offset: 平移偏移 {"x": int, "y": int}
            - alignment_similarity: 對齊後的相似度
            - error: 錯誤訊息（如果失敗）
        """
        service_start_time = time.time()
        logger.info(f"[服務層] 開始執行多印鑑比對服務")
        logger.info(f"[服務層] 圖像1 ID: {image1_id}")
        logger.info(f"[服務層] 印鑑數量: {len(seal_image_ids)}")
        logger.info(f"[服務層] 相似度閾值: {threshold}")
        logger.info(f"[服務層] 相似度權重 - SSIM: {similarity_ssim_weight}, Template: {similarity_template_weight}, Pixel: {pixel_similarity_weight}, Histogram: {histogram_similarity_weight}")
        logger.info(f"[服務層] Mask相似度權重 - 重疊區域: {overlap_weight}, 像素差異懲罰: {pixel_diff_penalty_weight}, 獨有區域懲罰: {unique_region_penalty_weight}")
        logger.info(f"[服務層] 圖像對齊參數 - 旋轉角度範圍: {rotation_range}度, 平移範圍: {translation_range}像素")
        
        try:
            # 獲取圖像1
            logger.info(f"[服務層] 獲取圖像1: {image1_id}")
            image1 = self.get_image(image1_id)
            if not image1:
                logger.error(f"[服務層] 圖像1不存在: {image1_id}")
                raise ImageNotFoundError("圖像1不存在")
            
            image1_path = Path(image1.file_path)
            if not image1_path.exists():
                logger.error(f"[服務層] 圖像1文件不存在: {image1_path}")
                raise ImageFileNotFoundError("圖像1文件不存在")
            
            logger.info(f"[服務層] 圖像1文件路徑: {image1_path}")
            
            # 檢查圖像1是否有標記印鑑位置
            if not image1.seal_bbox:
                logger.error(f"[服務層] 圖像1未標記印鑑位置: {image1_id}")
                raise ImageNotMarkedError("圖像1未標記印鑑位置，無法裁切")
            
            logger.info(f"[服務層] 圖像1印鑑位置: {image1.seal_bbox}")
            
            # 讀取原始圖像1
            image1_original = cv2.imread(str(image1_path))
            if image1_original is None:
                raise ImageReadError("無法讀取圖像1文件")
            
            # 獲取 bbox
            bbox1 = image1.seal_bbox
            
            # 計算裁切區域（不添加邊距）
            h, w = image1_original.shape[:2]
            x = max(0, bbox1['x'])
            y = max(0, bbox1['y'])
            crop_width = min(w - x, bbox1['width'])
            crop_height = min(h - y, bbox1['height'])
            
            # 確保不超出邊界
            if x + crop_width > w:
                crop_width = w - x
            if y + crop_height > h:
                crop_height = h - y
            
            if crop_width < 10 or crop_height < 10:
                raise CropAreaTooSmallError("裁切區域太小")
            
            # 裁切圖像
            image1_cropped = image1_original[y:y+crop_height, x:x+crop_width]
            
            # 保存裁切後的圖像
            original_name = Path(image1.filename).stem
            extension = Path(image1.filename).suffix or '.jpg'
            cropped_filename = f"{original_name}_cropped{extension}"
            cropped_file_path = self.upload_dir / cropped_filename
            
            # 確保文件名唯一
            counter = 1
            while cropped_file_path.exists():
                cropped_filename = f"{original_name}_cropped_{counter}{extension}"
                cropped_file_path = self.upload_dir / cropped_filename
                counter += 1
            
            cv2.imwrite(str(cropped_file_path), image1_cropped)
            
            # 創建新的圖像記錄
            image1_cropped_db = Image(
                filename=cropped_filename,
                file_path=str(cropped_file_path),
                file_size=get_file_size(cropped_file_path),
                mime_type=image1.mime_type
            )
            self.db.add(image1_cropped_db)
            self.db.commit()
            self.db.refresh(image1_cropped_db)
            
            # 使用裁切後的圖像路徑
            image1_cropped_path = cropped_file_path
            
            # 創建比對結果目錄
            comparison_dir = Path(settings.LOGS_DIR) / "multi_seal_comparisons"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            
            # 預載入所有印鑑圖像資訊（線程安全：在主線程中完成資料庫查詢）
            seal_images_data = {}
            for seal_image_id in seal_image_ids:
                seal_image = self.get_image(seal_image_id)
                if seal_image:
                    seal_image_path = Path(seal_image.file_path)
                    if seal_image_path.exists():
                        seal_images_data[seal_image_id] = {
                            'file_path': seal_image_path,
                            'image': seal_image
                        }
        
            # 準備線程池
            max_workers = min(len(seal_image_ids), settings.MAX_COMPARISON_THREADS)
            results = []
            results_lock = threading.Lock()
            
            logger.info(f"[服務層] 準備使用 {max_workers} 個線程進行並行比對")
            logger.info(f"[服務層] 預載入的印鑑圖像數量: {len(seal_images_data)}")
            comparison_start_time = time.time()
            
            def process_seal(seal_data):
                """處理單個印鑑的比對（線程函數）"""
                idx, seal_image_id = seal_data
                seal_start_time = time.time()
                logger.info(f"[服務層] 開始處理印鑑 {idx + 1}/{len(seal_image_ids)} (ID: {seal_image_id})")
                result = None
                try:
                    result = self._compare_single_seal(
                        image1_cropped_path=image1_cropped_path,
                        seal_image_id=seal_image_id,
                        seal_index=idx + 1,
                        image1_id=image1_id,
                        comparison_dir=comparison_dir,
                        threshold=threshold,
                        similarity_ssim_weight=similarity_ssim_weight,
                        similarity_template_weight=similarity_template_weight,
                        pixel_similarity_weight=pixel_similarity_weight,
                        histogram_similarity_weight=histogram_similarity_weight,
                        overlap_weight=overlap_weight,
                        pixel_diff_penalty_weight=pixel_diff_penalty_weight,
                        unique_region_penalty_weight=unique_region_penalty_weight,
                        rotation_range=rotation_range,
                        translation_range=translation_range,
                        seal_image_path=seal_images_data.get(seal_image_id, {}).get('file_path')
                    )
                    seal_elapsed = time.time() - seal_start_time
                    logger.info(f"[服務層] 印鑑 {idx + 1} 比對完成，耗時: {seal_elapsed:.2f} 秒，相似度: {result.get('similarity', 'N/A')}")
                except Exception as e:
                    # 如果 _compare_single_seal 拋出未預期的異常，創建錯誤結果
                    error_trace = traceback.format_exc()
                    logger.error(f"[服務層] 印鑑 {idx + 1} 比對時發生未預期錯誤: {e}")
                    logger.error(f"[服務層] 錯誤堆疊:\n{error_trace}")
                    result = {
                        'seal_index': idx + 1,
                        'seal_image_id': seal_image_id,
                        'similarity': None,
                        'is_match': None,
                        'overlay1_path': None,
                        'overlay2_path': None,
                        'heatmap_path': None,
                        'timing': {},  # 添加時間追蹤數據
                        'overlap_mask_path': None,
                        'pixel_diff_mask_path': None,
                        'diff_mask_2_only_path': None,
                        'diff_mask_1_only_path': None,
                        'mask_statistics': None,
                        'mask_based_similarity': None,
                        'input_image1_path': None,
                        'input_image2_path': None,
                        'alignment_angle': None,
                        'alignment_offset': None,
                        'alignment_similarity': None,
                        'alignment_success': False,
                        'error': f"比對過程發生未預期錯誤: {str(e)}"
                    }
                    seal_elapsed = time.time() - seal_start_time
                    logger.warning(f"[服務層] 印鑑 {idx + 1} 比對失敗，耗時: {seal_elapsed:.2f} 秒")
                
                # 確保結果被添加到列表（無論成功或失敗）
                with results_lock:
                    results.append(result)
                
                # 如果有回調函數，立即更新任務記錄
                if task_update_callback and result:
                    try:
                        task_update_callback(result, idx + 1, len(seal_image_ids))
                    except Exception as callback_error:
                        logger.error(f"[服務層] 更新任務記錄時出錯: {callback_error}")
        
            # 使用線程池並行處理
            logger.info(f"[服務層] 開始並行比對處理")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_seal, (idx, seal_id))
                    for idx, seal_id in enumerate(seal_image_ids)
                ]
                # 等待所有任務完成
                # 注意：process_seal 內部已經處理了所有異常並返回結果，這裡不需要額外的異常處理
                # 但如果 future 本身有問題（比如線程池關閉），仍然需要處理
                for future in as_completed(futures):
                    try:
                        future.result()  # 觸發異常處理（process_seal 內部已處理，這裡主要處理線程池級別的異常）
                    except Exception as e:
                        error_trace = traceback.format_exc()
                        logger.error(f"[服務層] 線程執行錯誤（線程池級別）: {e}")
                        logger.error(f"[服務層] 錯誤堆疊:\n{error_trace}")
                        # 注意：如果 future 本身失敗，process_seal 可能沒有執行，結果可能缺失
                        # 但這種情況很少見，通常是線程池配置問題
        
            comparison_elapsed = time.time() - comparison_start_time
            service_elapsed = time.time() - service_start_time
            
            # 按 seal_index 排序結果
            results.sort(key=lambda x: x.get('seal_index', 0))
            
            # 驗證結果數量是否正確
            expected_count = len(seal_image_ids)
            actual_count = len(results)
            if actual_count != expected_count:
                logger.warning(f"[服務層] 警告：結果數量不匹配！預期 {expected_count} 個，實際 {actual_count} 個")
                # 檢查缺失的印鑑索引
                result_indices = {r.get('seal_index') for r in results}
                expected_indices = set(range(1, expected_count + 1))
                missing_indices = expected_indices - result_indices
                if missing_indices:
                    logger.warning(f"[服務層] 缺失的印鑑索引: {sorted(missing_indices)}")
                    # 為缺失的印鑑創建錯誤結果
                    for missing_idx in sorted(missing_indices):
                        missing_seal_id = seal_image_ids[missing_idx - 1]
                        error_result = {
                            'seal_index': missing_idx,
                            'seal_image_id': missing_seal_id,
                            'similarity': None,
                            'is_match': None,
                            'overlay1_path': None,
                            'overlay2_path': None,
                            'heatmap_path': None,
                            'overlap_mask_path': None,
                            'pixel_diff_mask_path': None,
                            'diff_mask_2_only_path': None,
                            'diff_mask_1_only_path': None,
                            'mask_statistics': None,
                            'mask_based_similarity': None,
                            'input_image1_path': None,
                            'input_image2_path': None,
                            'alignment_angle': None,
                            'alignment_offset': None,
                            'alignment_similarity': None,
                            'alignment_success': False,
                            'error': "比對結果缺失（線程執行可能失敗）"
                        }
                        results.append(error_result)
                    # 重新排序
                    results.sort(key=lambda x: x.get('seal_index', 0))
            
            success_count = sum(1 for r in results if r.get('error') is None)
            failure_count = sum(1 for r in results if r.get('error') is not None)
            logger.info(f"[服務層] 多印鑑比對完成")
            logger.info(f"[服務層] 比對階段耗時: {comparison_elapsed:.2f} 秒 ({comparison_elapsed/60:.2f} 分鐘)")
            logger.info(f"[服務層] 總服務時間: {service_elapsed:.2f} 秒 ({service_elapsed/60:.2f} 分鐘)")
            logger.info(f"[服務層] 平均每個印鑑比對時間: {comparison_elapsed/len(seal_image_ids):.2f} 秒")
            if max_workers > 1:
                logger.info(f"[服務層] 理論加速比: {len(seal_image_ids) * (comparison_elapsed/len(seal_image_ids)) / comparison_elapsed:.2f}x")
            logger.info(f"[服務層] 比對結果統計: 總數={len(results)}, 成功={success_count}, 失敗={failure_count}")
            
            # 計算任務級別的時間數據
            task_timing = {
                'total_time': service_elapsed,  # 總服務時間
                'parallel_processing_time': comparison_elapsed,  # 並行處理時間
                'average_seal_time': comparison_elapsed / len(seal_image_ids) if len(seal_image_ids) > 0 else 0.0  # 平均每個印鑑時間
            }
            
            # 從各個結果中提取時間數據並計算平均值
            valid_timings = [r.get('timing', {}) for r in results if r.get('timing') and r.get('error') is None]
            if valid_timings:
                # 計算各個步驟的平均時間
                step_keys = ['load_images', 'remove_bg_image1', 'remove_bg_align_image2', 'save_aligned_images',
                           'similarity_calculation', 'save_corrected_images', 'create_overlay', 
                           'calculate_mask_stats', 'create_heatmap']
                for step_key in step_keys:
                    step_times = [t.get(step_key, 0.0) for t in valid_timings if step_key in t]
                    if step_times:
                        task_timing[f'avg_{step_key}'] = sum(step_times) / len(step_times)
                
                # 計算平均總時間（從各個結果的 total 字段）
                total_times = [t.get('total', 0.0) for t in valid_timings if 'total' in t]
                if total_times:
                    task_timing['avg_seal_total_time'] = sum(total_times) / len(total_times)
            
            # 返回結果和任務級別時間數據
            return {
                'results': results,
                'task_timing': task_timing
            }
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"[服務層] 多印鑑比對服務發生未預期錯誤")
            logger.error(f"[服務層] 錯誤類型: {type(e).__name__}")
            logger.error(f"[服務層] 錯誤訊息: {str(e)}")
            logger.error(f"[服務層] 錯誤堆疊:\n{error_trace}")
            raise
    
    def retry_missing_seals(
        self,
        image1_id: UUID,
        missing_seal_image_ids: List[UUID],
        missing_seal_indices: List[int],
        threshold: float,
        similarity_ssim_weight: float,
        similarity_template_weight: float,
        pixel_similarity_weight: float,
        histogram_similarity_weight: float,
        overlap_weight: float = 0.5,
        pixel_diff_penalty_weight: float = 0.3,
        unique_region_penalty_weight: float = 0.2,
        rotation_range: float = 15.0,
        translation_range: int = 100
    ) -> List[Dict]:
        """
        重新比對缺失的印鑑（單線程執行，避免數據庫會話衝突）
        
        Args:
            image1_id: 圖像1 ID
            missing_seal_image_ids: 缺失的印鑑圖像 ID 列表
            missing_seal_indices: 缺失的印鑑索引列表（從1開始）
            threshold: 相似度閾值
            similarity_ssim_weight: SSIM 權重
            similarity_template_weight: 模板匹配權重
            pixel_similarity_weight: 像素相似度權重
            histogram_similarity_weight: 直方圖相似度權重
            
        Returns:
            比對結果列表
        """
        logger.info(f"[服務層] 開始重新比對缺失的印鑑")
        logger.info(f"[服務層] 缺失印鑑數量: {len(missing_seal_image_ids)}")
        logger.info(f"[服務層] 缺失印鑑索引: {missing_seal_indices}")
        
        retry_results = []
        retry_start_time = time.time()
        
        try:
            # 獲取圖像1
            image1 = self.get_image(image1_id)
            if not image1:
                logger.error(f"[服務層] 圖像1不存在: {image1_id}")
                raise ImageNotFoundError("圖像1不存在")
            
            image1_path = Path(image1.file_path)
            if not image1_path.exists():
                logger.error(f"[服務層] 圖像1文件不存在: {image1_path}")
                raise ImageFileNotFoundError("圖像1文件不存在")
            
            # 檢查圖像1是否有標記印鑑位置
            if not image1.seal_bbox:
                logger.error(f"[服務層] 圖像1未標記印鑑位置: {image1_id}")
                raise ImageNotMarkedError("圖像1未標記印鑑位置，無法裁切")
            
            # 讀取原始圖像1
            image1_original = cv2.imread(str(image1_path))
            if image1_original is None:
                raise ImageReadError("無法讀取圖像1文件")
            
            # 獲取 bbox
            bbox1 = image1.seal_bbox
            
            # 計算裁切區域
            h, w = image1_original.shape[:2]
            x = max(0, bbox1['x'])
            y = max(0, bbox1['y'])
            crop_width = min(w - x, bbox1['width'])
            crop_height = min(h - y, bbox1['height'])
            
            if x + crop_width > w:
                crop_width = w - x
            if y + crop_height > h:
                crop_height = h - y
            
            if crop_width < 10 or crop_height < 10:
                raise CropAreaTooSmallError("裁切區域太小")
            
            # 裁切圖像
            image1_cropped = image1_original[y:y+crop_height, x:x+crop_width]
            
            # 保存裁切後的圖像（重用現有的裁切圖像，如果存在）
            original_name = Path(image1.filename).stem
            extension = Path(image1.filename).suffix or '.jpg'
            cropped_filename = f"{original_name}_cropped{extension}"
            cropped_file_path = self.upload_dir / cropped_filename
            
            # 確保文件名唯一
            counter = 1
            while cropped_file_path.exists():
                cropped_filename = f"{original_name}_cropped_{counter}{extension}"
                cropped_file_path = self.upload_dir / cropped_filename
                counter += 1
            
            # 如果文件不存在，創建它
            if not cropped_file_path.exists():
                cv2.imwrite(str(cropped_file_path), image1_cropped)
            
            image1_cropped_path = cropped_file_path
            
            # 創建比對結果目錄
            comparison_dir = Path(settings.LOGS_DIR) / "multi_seal_comparisons"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            
            # 預載入缺失的印鑑圖像資訊
            seal_images_data = {}
            for seal_image_id in missing_seal_image_ids:
                seal_image = self.get_image(seal_image_id)
                if seal_image:
                    seal_image_path = Path(seal_image.file_path)
                    if seal_image_path.exists():
                        seal_images_data[seal_image_id] = {
                            'file_path': seal_image_path,
                            'image': seal_image
                        }
                    else:
                        logger.warning(f"[服務層] 印鑑圖像文件不存在: {seal_image_path}")
                else:
                    logger.warning(f"[服務層] 印鑑圖像不存在: {seal_image_id}")
            
            # 單線程順序執行比對（避免數據庫會話衝突）
            for idx, (seal_image_id, seal_index) in enumerate(zip(missing_seal_image_ids, missing_seal_indices)):
                logger.info(f"[服務層] 重新比對印鑑 {seal_index} (ID: {seal_image_id}) [{idx + 1}/{len(missing_seal_image_ids)}]")
                seal_start_time = time.time()
                
                try:
                    result = self._compare_single_seal(
                        image1_cropped_path=image1_cropped_path,
                        seal_image_id=seal_image_id,
                        seal_index=seal_index,
                        image1_id=image1_id,
                        comparison_dir=comparison_dir,
                        threshold=threshold,
                        similarity_ssim_weight=similarity_ssim_weight,
                        similarity_template_weight=similarity_template_weight,
                        pixel_similarity_weight=pixel_similarity_weight,
                        histogram_similarity_weight=histogram_similarity_weight,
                        overlap_weight=overlap_weight,
                        pixel_diff_penalty_weight=pixel_diff_penalty_weight,
                        unique_region_penalty_weight=unique_region_penalty_weight,
                        rotation_range=rotation_range,
                        translation_range=translation_range,
                        seal_image_path=seal_images_data.get(seal_image_id, {}).get('file_path')
                    )
                    seal_elapsed = time.time() - seal_start_time
                    logger.info(f"[服務層] 印鑑 {seal_index} 重新比對完成，耗時: {seal_elapsed:.2f} 秒，相似度: {result.get('similarity', 'N/A')}")
                    retry_results.append(result)
                except Exception as e:
                    error_trace = traceback.format_exc()
                    logger.error(f"[服務層] 印鑑 {seal_index} 重新比對時發生錯誤: {e}")
                    logger.error(f"[服務層] 錯誤堆疊:\n{error_trace}")
                    result = {
                        'seal_index': seal_index,
                        'seal_image_id': seal_image_id,
                        'similarity': None,
                        'is_match': None,
                        'overlay1_path': None,
                        'overlay2_path': None,
                        'heatmap_path': None,
                        'overlap_mask_path': None,
                        'pixel_diff_mask_path': None,
                        'diff_mask_2_only_path': None,
                        'diff_mask_1_only_path': None,
                        'mask_statistics': None,
                        'mask_based_similarity': None,
                        'input_image1_path': None,
                        'input_image2_path': None,
                        'alignment_angle': None,
                        'alignment_offset': None,
                        'alignment_similarity': None,
                        'alignment_success': False,
                        'error': f"重新比對失敗: {str(e)}",
                        'timing': {}  # 添加時間追蹤數據
                    }
                    retry_results.append(result)
            
            retry_elapsed = time.time() - retry_start_time
            success_count = sum(1 for r in retry_results if r.get('error') is None)
            logger.info(f"[服務層] 重新比對完成，耗時: {retry_elapsed:.2f} 秒")
            logger.info(f"[服務層] 重新比對結果統計: 總數={len(retry_results)}, 成功={success_count}, 失敗={len(retry_results) - success_count}")
            
            return retry_results
            
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"[服務層] 重新比對缺失印鑑時發生未預期錯誤")
            logger.error(f"[服務層] 錯誤類型: {type(e).__name__}")
            logger.error(f"[服務層] 錯誤訊息: {str(e)}")
            logger.error(f"[服務層] 錯誤堆疊:\n{error_trace}")
            # 為所有缺失的印鑑創建錯誤結果
            for seal_image_id, seal_index in zip(missing_seal_image_ids, missing_seal_indices):
                retry_results.append({
                    'seal_index': seal_index,
                    'seal_image_id': seal_image_id,
                    'similarity': None,
                    'is_match': None,
                    'overlay1_path': None,
                    'overlay2_path': None,
                    'heatmap_path': None,
                    'overlap_mask_path': None,
                    'pixel_diff_mask_path': None,
                    'diff_mask_2_only_path': None,
                    'diff_mask_1_only_path': None,
                    'mask_statistics': None,
                    'mask_based_similarity': None,
                    'input_image1_path': None,
                    'input_image2_path': None,
                    'alignment_angle': None,
                    'alignment_offset': None,
                    'alignment_similarity': None,
                    'alignment_success': False,
                    'error': f"重新比對過程發生未預期錯誤: {str(e)}"
                })
            return retry_results
    
    def _compare_single_seal(
        self,
        image1_cropped_path: Path,
        seal_image_id: UUID,
        seal_index: int,
        image1_id: UUID,
        comparison_dir: Path,
        threshold: float,
        similarity_ssim_weight: float,
        similarity_template_weight: float,
        pixel_similarity_weight: float,
        histogram_similarity_weight: float,
        overlap_weight: float = 0.5,
        pixel_diff_penalty_weight: float = 0.3,
        unique_region_penalty_weight: float = 0.2,
        rotation_range: float = 15.0,
        translation_range: int = 100,
        seal_image_path: Optional[Path] = None
    ) -> Dict:
        """
        單個印鑑的比對邏輯（線程安全版本）
        
        注意：is_match 的判定基於 mask_based_similarity 與 threshold 的比較，
        而不是傳統的相似度指標。如果 mask 相似度計算失敗，is_match 將設為 None。
        
        Args:
            image1_cropped_path: 裁切後的圖像1路徑
            seal_image_id: 印鑑圖像 ID
            seal_index: 印鑑索引（從1開始）
            image1_id: 圖像1 ID
            comparison_dir: 比對結果目錄
            threshold: 相似度閾值（用於判定 mask_based_similarity 是否匹配）
            similarity_ssim_weight: SSIM 權重（用於傳統相似度計算，僅用於顯示）
            similarity_template_weight: 模板匹配權重（用於傳統相似度計算，僅用於顯示）
            pixel_similarity_weight: 像素相似度權重（用於傳統相似度計算，僅用於顯示）
            histogram_similarity_weight: 直方圖相似度權重（用於傳統相似度計算，僅用於顯示）
            seal_image_path: 印鑑圖像路徑（預載入，避免線程中使用資料庫）
            
        Returns:
            比對結果字典，包含：
            - similarity: 傳統相似度（用於顯示）
            - mask_based_similarity: 基於mask的相似度（用於判定is_match）
            - is_match: 是否匹配（基於mask_based_similarity >= threshold判定）
            - 如果mask相似度計算失敗，is_match為None
        """
        result = {
            'seal_index': seal_index,
            'seal_image_id': seal_image_id,
            'similarity': None,
            'is_match': None,
            'overlay1_path': None,
            'overlay2_path': None,
            'heatmap_path': None,
            'overlap_mask_path': None,
            'pixel_diff_mask_path': None,
            'diff_mask_2_only_path': None,
            'diff_mask_1_only_path': None,
            'mask_statistics': None,
            'mask_based_similarity': None,
            'input_image1_path': None,
            'input_image2_path': None,
            'alignment_angle': None,
            'alignment_offset': None,
            'alignment_similarity': None,
            'alignment_success': False,
            'error': None,
            'timing': {}  # 時間追蹤數據
        }
        
        # 初始化時間追蹤
        timing = {}
        step_start_time = time.time()
        total_start_time = step_start_time
        
        try:
            # 檢查印鑑圖像路徑
            if not seal_image_path or not seal_image_path.exists():
                result['error'] = "印鑑圖像文件不存在"
                return result
            
            # 執行比對
            # 使用裁切後的圖像1和圖像2進行比對
            # 流程：載入圖像 → 去背景 → 旋轉對齊 → 保存文件 → 計算相似度
            
            # 生成記錄 ID（用於文件命名）
            record_id = f"{image1_id}_{seal_image_id}_{seal_index}"
            
            # 1. 載入圖像
            step_start = time.time()
            img1_original = cv2.imread(str(image1_cropped_path))
            img2_original = cv2.imread(str(seal_image_path))
            timing['load_images'] = time.time() - step_start
            
            if img1_original is None:
                result['error'] = "無法載入圖像1"
                result['timing'] = timing
                return result
            
            if img2_original is None:
                result['error'] = "無法載入印鑑圖像"
                result['timing'] = timing
                return result
            
            # 2. 去背景處理（圖像1）
            step_start = time.time()
            img1_no_bg, _, _, _, _, _ = self._remove_background_and_align(img1_original.copy())
            timing['remove_bg_image1'] = time.time() - step_start
            
            # 3. 去背景和對齊處理（圖像2，相對於圖像1）
            step_start = time.time()
            img2_aligned, alignment_angle, alignment_offset, alignment_similarity, alignment_metrics, alignment_timing = self._remove_background_and_align(
                img2_original.copy(),
                reference_image=img1_no_bg,  # 傳入去背景後的圖像1作為參考
                is_image2=True,
                rotation_range=rotation_range,
                translation_range=translation_range
            )
            timing['remove_bg_align_image2'] = time.time() - step_start
            # 將對齊各階段時間存儲到 timing 中
            if alignment_timing:
                timing['alignment_stages'] = alignment_timing
            
            # 記錄對齊結果
            if alignment_angle is not None and alignment_offset is not None:
                result['alignment_success'] = True
                result['alignment_angle'] = float(alignment_angle)
                result['alignment_offset'] = {
                    'x': int(alignment_offset[0]),
                    'y': int(alignment_offset[1])
                }
                if alignment_similarity is not None:
                    result['alignment_similarity'] = float(alignment_similarity)
                # 回傳精簡的 alignment_metrics（避免 payload 過大，但保留 debug 關鍵資訊）
                if alignment_metrics and isinstance(alignment_metrics, dict):
                    allow_keys = {
                        'coarse_search_mode',
                        'coarse_similarity',
                        'multistart_triggered',
                        'multistart_best_overlap',
                        'multistart_best_candidate',
                        'multistart_adopted',
                        'rescue_triggered',
                        'rescue_overlap_before',
                        'rescue_overlap_after',
                        'overlap_after_stage12',
                        'overlap_after_stage3',
                        'overlap_after_stage4',
                        'offset_after_stage12',
                        'offset_after_stage3',
                        'offset_after_stage4',
                        'post_fine_rescue_triggered',
                        'post_fine_rescue_overlap_before',
                        'post_fine_rescue_overlap_after',
                        'post_fine_rescue_offset',
                        'post_fine_rescue_overlap_final',
                        'angle_sign_check_triggered',
                        'angle_sign_flipped',
                        'overlap_before_sign_check',
                        'overlap_after_sign_check',
                        'final_angle',
                        'final_offset_x',
                        'final_offset_y',
                        'final_similarity',
                        'alignment_canvas',
                        'alignment_canvas_mode'
                    }
                    result['alignment_metrics'] = {k: alignment_metrics.get(k) for k in allow_keys if k in alignment_metrics}
                similarity_str = f"{alignment_similarity:.4f}" if alignment_similarity is not None else 'N/A'
                print(f"印鑑 {seal_index} 對齊成功: 角度={alignment_angle:.2f}度, 偏移=({alignment_offset[0]}, {alignment_offset[1]}), 相似度={similarity_str}")
            else:
                result['alignment_success'] = False
                print(f"警告：印鑑 {seal_index} 對齊失敗，使用去背景後的圖像（未對齊）")
            
            # 4. 保存對齊後的圖像到文件（在比對之前）
            step_start = time.time()
            try:
                image1_for_comparison = comparison_dir / f"image1_cropped_{record_id}.jpg"
                # 若對齊使用 auto-canvas（避免裁切），需要把 image1 也補到同一畫布與同一 shift
                img1_to_save = img1_no_bg
                try:
                    if alignment_metrics and isinstance(alignment_metrics, dict):
                        canvas = alignment_metrics.get('alignment_canvas')
                        if canvas and isinstance(canvas, dict):
                            shift_x = int(canvas.get('shift_x', 0))
                            shift_y = int(canvas.get('shift_y', 0))
                            canvas_w = int(canvas.get('w', img1_no_bg.shape[1]))
                            canvas_h = int(canvas.get('h', img1_no_bg.shape[0]))

                            if canvas_w > 0 and canvas_h > 0 and (shift_x != 0 or shift_y != 0):
                                if img1_no_bg.ndim == 2:
                                    padded = np.full((canvas_h, canvas_w), 255, dtype=img1_no_bg.dtype)
                                else:
                                    padded = np.full((canvas_h, canvas_w, img1_no_bg.shape[2]), 255, dtype=img1_no_bg.dtype)

                                src_h, src_w = img1_no_bg.shape[:2]
                                x_end = min(canvas_w, shift_x + src_w)
                                y_end = min(canvas_h, shift_y + src_h)
                                if x_end > shift_x and y_end > shift_y:
                                    padded[shift_y:y_end, shift_x:x_end] = img1_no_bg[0:(y_end - shift_y), 0:(x_end - shift_x)]
                                    img1_to_save = padded
                except Exception as _pad_e:
                    # padding 失敗不應阻斷流程（仍可用原圖繼續）
                    img1_to_save = img1_no_bg

                if not cv2.imwrite(str(image1_for_comparison), img1_to_save):
                    raise IOError(f"無法保存圖像1到 {image1_for_comparison}")
                
                image2_for_comparison = comparison_dir / f"image2_cropped_{record_id}.jpg"
                if not cv2.imwrite(str(image2_for_comparison), img2_aligned):
                    raise IOError(f"無法保存圖像2到 {image2_for_comparison}")
                
                # 保存輸入圖像路徑到結果中（疊圖前的圖像）
                result['input_image1_path'] = image1_for_comparison.name
                result['input_image2_path'] = image2_for_comparison.name
                timing['save_aligned_images'] = time.time() - step_start
            except Exception as e:
                timing['save_aligned_images'] = time.time() - step_start
                result['error'] = f"保存對齊後的圖像失敗: {str(e)}"
                result['timing'] = timing
                print(f"錯誤：保存對齊後的圖像失敗 (印鑑 {seal_index}): {e}")
                import traceback
                traceback.print_exc()
                return result
            
            # 5. 創建獨立的 comparator 實例（線程安全）
            comparator = SealComparator(
                threshold=threshold,
                similarity_ssim_weight=similarity_ssim_weight,
                similarity_template_weight=similarity_template_weight,
                pixel_similarity_weight=pixel_similarity_weight,
                histogram_similarity_weight=histogram_similarity_weight
            )
            
            # 6. 使用 compare_files 比對（傳入文件路徑）
            step_start = time.time()
            try:
                # 確保文件存在
                if not image1_for_comparison.exists():
                    raise FileNotFoundError(f"圖像1文件不存在: {image1_for_comparison}")
                if not image2_for_comparison.exists():
                    raise FileNotFoundError(f"圖像2文件不存在: {image2_for_comparison}")
                
                is_match, similarity, details, img2_corrected, img1_corrected = comparator.compare_files(
                    str(image1_for_comparison),
                    str(image2_for_comparison),
                    enable_rotation_search=False,  # 已經對齊，不需要旋轉搜索
                    enable_translation_search=False,  # 已經對齊，不需要平移搜索
                    bbox1=None,  # 已經裁切並對齊好了
                    bbox2=None   # 已經裁切並對齊好了
                )
                
                # 保留傳統相似度用於顯示，但is_match將基於mask相似度判定
                result['similarity'] = float(similarity)
                # 注意：is_match將在計算完mask相似度後設定
                timing['similarity_calculation'] = time.time() - step_start
            except Exception as e:
                timing['similarity_calculation'] = time.time() - step_start
                result['error'] = f"相似度計算失敗: {str(e)}"
                result['timing'] = timing
                print(f"錯誤：相似度計算失敗 (印鑑 {seal_index}): {e}")
                import traceback
                traceback.print_exc()
                return result
            
            # 7. 保存校正後的圖像（使用 compare_files 返回的圖像）
            step_start = time.time()
            image1_corrected_path = None
            image2_corrected_path = None
            
            if img1_corrected is not None:
                try:
                    corrected_file1 = comparison_dir / f"corrected_image1_{record_id}.jpg"
                    cv2.imwrite(str(corrected_file1), img1_corrected)
                    image1_corrected_path = str(corrected_file1)
                except Exception as e:
                    print(f"警告：保存校正後的圖像1失敗 (印鑑 {seal_index}): {e}")
                    image1_corrected_path = str(image1_for_comparison)
            else:
                # 如果 compare_files 沒有返回 img1_corrected，使用已保存的去背景圖像
                image1_corrected_path = str(image1_for_comparison)
            
            if img2_corrected is not None:
                try:
                    corrected_file2 = comparison_dir / f"corrected_{record_id}.jpg"
                    cv2.imwrite(str(corrected_file2), img2_corrected)
                    image2_corrected_path = str(corrected_file2)
                except Exception as e:
                    print(f"警告：保存校正後的圖像2失敗 (印鑑 {seal_index}): {e}")
                    image2_corrected_path = str(image2_for_comparison)
            else:
                # 如果 compare_files 沒有返回 img2_corrected，使用已保存的對齊圖像
                image2_corrected_path = str(image2_for_comparison)
            timing['save_corrected_images'] = time.time() - step_start
            
            # 8. 生成疊圖（使用對齊後的圖像）
            step_start = time.time()
            try:
                # 使用對齊後的圖像1和圖像2生成疊圖
                # 優先使用 compare_files 返回的校正圖像，否則使用已保存的對齊圖像
                image1_for_overlay = str(image1_corrected_path) if image1_corrected_path else str(image1_for_comparison)
                image2_for_overlay = str(image2_corrected_path) if image2_corrected_path else str(image2_for_comparison)
                
                # 確保圖像路徑存在
                if not Path(image1_for_overlay).exists():
                    print(f"警告：印鑑 {seal_index} 的圖像1路徑不存在: {image1_for_overlay}，跳過疊圖生成")
                    # 無法生成mask，is_match設為None
                    if 'is_match' not in result or result.get('is_match') is None:
                        result['is_match'] = None
                    timing['create_overlay'] = time.time() - step_start
                elif not Path(image2_for_overlay).exists():
                    print(f"警告：印鑑 {seal_index} 的圖像2路徑不存在: {image2_for_overlay}，跳過疊圖生成")
                    # 無法生成mask，is_match設為None
                    if 'is_match' not in result or result.get('is_match') is None:
                        result['is_match'] = None
                    timing['create_overlay'] = time.time() - step_start
                else:
                    # 使用對齊後的圖像生成疊圖
                    # image1_path: 對齊後的圖像1
                    # image2_path: 對齊後的圖像2（作為原始圖像2）
                    # image2_corrected_path: 不需要，因為已經使用對齊後的圖像
                    overlay1_path, overlay2_path, overlap_mask_path, pixel_diff_mask_path, diff_mask_2_only_path, diff_mask_1_only_path, gray_diff_path = create_overlay_image(
                        image1_for_overlay,
                        image2_for_overlay,  # 使用對齊後的圖像2作為原始圖像2
                        comparison_dir,
                        record_id,
                        image2_corrected_path=None  # 不需要，因為已經使用對齊後的圖像
                    )
                    timing['create_overlay'] = time.time() - step_start
                    if overlay1_path and Path(overlay1_path).exists():
                        # 只保存文件名，前端通過 API 獲取
                        result['overlay1_path'] = Path(overlay1_path).name
                        print(f"印鑑 {seal_index} 疊圖1生成成功: {result['overlay1_path']}")
                    else:
                        print(f"警告：印鑑 {seal_index} 疊圖1生成失敗或文件不存在")
                    
                    if overlay2_path and Path(overlay2_path).exists():
                        result['overlay2_path'] = Path(overlay2_path).name
                        print(f"印鑑 {seal_index} 疊圖2生成成功: {result['overlay2_path']}")
                    else:
                        print(f"警告：印鑑 {seal_index} 疊圖2生成失敗或文件不存在")
                    
                    # 保存mask路徑和gray_diff路徑
                    if overlap_mask_path and Path(overlap_mask_path).exists():
                        result['overlap_mask_path'] = Path(overlap_mask_path).name
                    if pixel_diff_mask_path and Path(pixel_diff_mask_path).exists():
                        result['pixel_diff_mask_path'] = Path(pixel_diff_mask_path).name
                    if diff_mask_2_only_path and Path(diff_mask_2_only_path).exists():
                        result['diff_mask_2_only_path'] = Path(diff_mask_2_only_path).name
                    if diff_mask_1_only_path and Path(diff_mask_1_only_path).exists():
                        result['diff_mask_1_only_path'] = Path(diff_mask_1_only_path).name
                    if gray_diff_path and Path(gray_diff_path).exists():
                        result['gray_diff_path'] = Path(gray_diff_path).name
                        print(f"印鑑 {seal_index} gray_diff視覺化生成成功: {result['gray_diff_path']}")
                    
                    # 計算mask統計資訊和基於mask的相似度
                    step_start = time.time()
                    try:
                        mask_stats = calculate_mask_statistics(
                            overlap_mask_path,
                            pixel_diff_mask_path,
                            diff_mask_2_only_path,
                            diff_mask_1_only_path
                        )
                        timing['calculate_mask_stats'] = time.time() - step_start
                        
                        # 檢查是否有有效的mask數據（total_seal_pixels > 0）
                        if mask_stats.get('total_seal_pixels', 0) == 0:
                            # 沒有有效的mask數據，無法計算相似度
                            result['is_match'] = None
                            print(f"警告：印鑑 {seal_index} 沒有有效的mask數據（total_seal_pixels=0），無法判定匹配狀態")
                        else:
                            mask_based_similarity = calculate_mask_based_similarity(
                                mask_stats,
                                overlap_weight=overlap_weight,
                                pixel_diff_penalty_weight=pixel_diff_penalty_weight,
                                unique_region_penalty_weight=unique_region_penalty_weight
                            )
                            result['mask_statistics'] = mask_stats
                            result['mask_based_similarity'] = float(mask_based_similarity)
                            
                            # 使用mask相似度與threshold比較來判定is_match
                            if mask_based_similarity is not None:
                                result['is_match'] = mask_based_similarity >= threshold
                                print(f"印鑑 {seal_index} mask-based相似度: {mask_based_similarity:.4f}, 判定結果: {'匹配' if result['is_match'] else '不匹配'} (threshold={threshold})")
                            else:
                                result['is_match'] = None
                                print(f"警告：印鑑 {seal_index} mask相似度為None，無法判定匹配狀態")
                    except Exception as e:
                        timing['calculate_mask_stats'] = time.time() - step_start
                        print(f"警告：計算mask統計資訊失敗 (印鑑 {seal_index}): {e}")
                        import traceback
                        traceback.print_exc()
                        # mask相似度計算失敗，is_match設為None
                        result['is_match'] = None
            except Exception as e:
                print(f"生成疊圖失敗 (印鑑 {seal_index}): {e}")
                import traceback
                traceback.print_exc()
                # 記錄錯誤但不阻止比對完成
                if 'overlay_error' not in result:
                    result['overlay_error'] = str(e)
                # overlay生成失敗，無法計算mask相似度，is_match設為None
                if 'is_match' not in result or result.get('is_match') is None:
                    result['is_match'] = None
            
            # 9. 生成熱力圖（使用對齊後的圖像）
            step_start = time.time()
            try:
                # create_difference_heatmap 需要 record_id 為 int，但我們使用字符串
                # 使用 hash 轉換為數字
                record_id_int = hash(record_id) % (10 ** 9)  # 轉換為正整數
                
                # 優先使用 compare_files 返回的校正圖像，否則使用已保存的對齊圖像
                image1_for_heatmap = str(image1_corrected_path) if image1_corrected_path else str(image1_for_comparison)
                image2_for_heatmap = str(image2_corrected_path) if image2_corrected_path else str(image2_for_comparison)
                
                # 確保 seal_image_path 不為 None
                if not seal_image_path:
                    print(f"警告：印鑑 {seal_index} 的圖像路徑為 None，跳過熱力圖生成")
                    timing['create_heatmap'] = time.time() - step_start
                else:
                    heatmap_path, _ = create_difference_heatmap(
                        image1_for_heatmap,  # 使用對齊後的圖像1
                        image2_for_heatmap,  # 使用對齊後的圖像2
                        str(seal_image_path),
                        comparison_dir,
                        record_id_int
                    )
                    timing['create_heatmap'] = time.time() - step_start
                    if heatmap_path:
                        # create_difference_heatmap 返回相對路徑 "heatmaps/heatmap_{record_id}.jpg"
                        # 但實際文件保存在 comparison_dir / f"heatmap_{record_id}.jpg"
                        # 直接使用實際保存的文件名
                        actual_heatmap_file = comparison_dir / f"heatmap_{record_id_int}.jpg"
                        if actual_heatmap_file.exists():
                            result['heatmap_path'] = actual_heatmap_file.name
                        else:
                            # 如果實際文件不存在，嘗試使用返回的相對路徑
                            if not Path(heatmap_path).is_absolute():
                                heatmap_path_full = comparison_dir / heatmap_path
                            else:
                                heatmap_path_full = Path(heatmap_path)
                            if heatmap_path_full.exists():
                                result['heatmap_path'] = heatmap_path_full.name
                            else:
                                print(f"警告：熱力圖文件不存在，返回路徑={heatmap_path}, 實際路徑={actual_heatmap_file}")
            except Exception as e:
                timing['create_heatmap'] = time.time() - step_start
                print(f"生成熱力圖失敗 (印鑑 {seal_index}): {e}")
                import traceback
                traceback.print_exc()
                # 不設置錯誤，繼續處理
            
            # 計算總時間
            timing['total'] = time.time() - total_start_time
            
            # 將時間數據添加到結果中
            result['timing'] = timing
            
        except Exception as e:
            # 即使出錯也要記錄已完成的步驟時間
            timing['total'] = time.time() - total_start_time
            result['timing'] = timing
            result['error'] = str(e)
            print(f"比對印鑑 {seal_index} 時出錯: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def detect_matching_seals_with_rotation(
        self,
        image1_id: UUID,
        image2_id: UUID,
        rotation_range: float = 15.0,
        angle_step: float = 1.0,
        max_seals: int = 10
    ) -> List[Dict]:
        """
        檢測圖像2中與圖像1最相似的印鑑（考慮旋轉）
        
        Args:
            image1_id: 參考圖像 ID（模板）
            image2_id: 包含多個印鑑的圖像 ID
            rotation_range: 旋轉角度範圍（度），默認45度
            angle_step: 旋轉角度步長（度），默認1度
            max_seals: 最大返回數量，默認10個
            
        Returns:
            匹配結果列表，每個元素包含：
            - bbox: 邊界框
            - center: 中心點
            - rotation_angle: 最佳旋轉角度
            - similarity: 相似度
            - confidence: 置信度
        """
        # 獲取圖像1
        image1 = self.get_image(image1_id)
        if not image1:
            raise ImageNotFoundError("圖像1不存在")
        
        image1_path = Path(image1.file_path)
        if not image1_path.exists():
            raise ImageFileNotFoundError("圖像1文件不存在")
        
        # 獲取圖像2
        image2 = self.get_image(image2_id)
        if not image2:
            raise ImageNotFoundError("圖像2不存在")
        
        image2_path = Path(image2.file_path)
        if not image2_path.exists():
            raise ImageFileNotFoundError("圖像2文件不存在")
        
        # 執行旋轉匹配檢測
        result = detect_seals_with_rotation_matching(
            str(image1_path),
            str(image2_path),
            rotation_range=rotation_range,
            angle_step=angle_step,
            max_seals=max_seals,
            timeout=30.0
        )
        
        if not result.get('detected'):
            return []
        
        return result.get('matches', [])
    
    def create_multi_seal_comparison_task(
        self,
        image1_id: UUID,
        seal_image_ids: List[UUID],
        threshold: float = 0.95,
        similarity_ssim_weight: float = 0.5,
        similarity_template_weight: float = 0.35,
        pixel_similarity_weight: float = 0.1,
        histogram_similarity_weight: float = 0.05
    ) -> MultiSealComparisonTask:
        """
        創建多印鑑比對任務
        
        Args:
            image1_id: 圖像1 ID
            seal_image_ids: 印鑑圖像 ID 列表
            threshold: 相似度閾值
            similarity_ssim_weight: SSIM 權重
            similarity_template_weight: 模板匹配權重
            pixel_similarity_weight: 像素相似度權重
            histogram_similarity_weight: 直方圖相似度權重
            
        Returns:
            MultiSealComparisonTask 模型實例
        """
        import uuid as uuid_lib
        
        task_uid = str(uuid_lib.uuid4())
        
        task = MultiSealComparisonTask(
            task_uid=task_uid,
            image1_id=image1_id,
            status=ComparisonStatus.PENDING,
            seal_image_ids=[str(sid) for sid in seal_image_ids],
            threshold=threshold,
            similarity_ssim_weight=similarity_ssim_weight,
            similarity_template_weight=similarity_template_weight,
            pixel_similarity_weight=pixel_similarity_weight,
            histogram_similarity_weight=histogram_similarity_weight
        )
        
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)
        
        return task
    
    def get_multi_seal_comparison_task(self, task_uid: str) -> Optional[MultiSealComparisonTask]:
        """
        獲取多印鑑比對任務
        
        Args:
            task_uid: 任務 UID
            
        Returns:
            MultiSealComparisonTask 模型實例或 None
        """
        return self.db.query(MultiSealComparisonTask).filter(
            MultiSealComparisonTask.task_uid == task_uid
        ).first()
    
    def get_multi_seal_comparison_task_or_raise(self, task_uid: str) -> MultiSealComparisonTask:
        """
        獲取多印鑑比對任務，如果不存在則拋出異常
        
        Args:
            task_uid: 任務 UID
            
        Returns:
            MultiSealComparisonTask 模型實例
            
        Raises:
            MultiSealComparisonTaskNotFoundError: 任務不存在
        """
        task = self.get_multi_seal_comparison_task(task_uid)
        if not task:
            raise MultiSealComparisonTaskNotFoundError("任務不存在")
        return task

