"""
圖像服務
"""

from pathlib import Path
from sqlalchemy.orm import Session
from typing import Optional
from uuid import UUID
import cv2

from app.models import Image
from app.schemas import ImageCreate, ImageResponse
from app.utils.image_utils import save_uploaded_file, get_file_size, delete_file
from app.utils.seal_detector import detect_seal_location, detect_multiple_seals, detect_seals_with_rotation_matching
import sys
from pathlib import Path as PathLib
core_path = PathLib(__file__).parent.parent.parent / "core"
sys.path.insert(0, str(core_path))
from seal_compare import SealComparator
from overlay import create_overlay_image
from verification import create_difference_heatmap
from fastapi import UploadFile, HTTPException
from app.config import settings
from typing import Dict, Optional, List, Tuple
import numpy as np


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
            raise HTTPException(status_code=404, detail="圖像不存在")
        
        file_path = Path(db_image.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="圖像文件不存在")
        
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
            raise HTTPException(status_code=404, detail="圖像不存在")
        
        # 驗證數據
        if bbox:
            if not all(k in bbox for k in ['x', 'y', 'width', 'height']):
                raise HTTPException(status_code=400, detail="邊界框格式錯誤")
            if bbox['width'] < 10 or bbox['height'] < 10:
                raise HTTPException(status_code=400, detail="邊界框尺寸太小")
        
        if center:
            if not all(k in center for k in ['center_x', 'center_y', 'radius']):
                raise HTTPException(status_code=400, detail="中心點格式錯誤")
        
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
    
    def detect_multiple_seals(self, image_id: UUID) -> Dict:
        """
        檢測圖像中的多個印鑑位置
        
        Args:
            image_id: 圖像 ID
            
        Returns:
            檢測結果字典
        """
        db_image = self.get_image(image_id)
        if not db_image:
            raise HTTPException(status_code=404, detail="圖像不存在")
        
        file_path = Path(db_image.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="圖像文件不存在")
        
        # 執行多印鑑檢測（帶超時保護）
        detection_result = detect_multiple_seals(str(file_path), timeout=5.0, max_seals=10)
        
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
            raise HTTPException(status_code=404, detail="圖像不存在")
        
        # 驗證數據格式
        normalized_seals = []
        for seal in seals:
            if 'bbox' not in seal or 'center' not in seal:
                raise HTTPException(status_code=400, detail="印鑑數據格式錯誤")
            
            bbox = seal['bbox']
            center = seal['center']
            confidence = seal.get('confidence', 0.5)
            
            # 驗證邊界框
            if not all(k in bbox for k in ['x', 'y', 'width', 'height']):
                raise HTTPException(status_code=400, detail="邊界框格式錯誤")
            if bbox['width'] < 10 or bbox['height'] < 10:
                continue  # 跳過太小的框
            
            # 驗證中心點
            if not all(k in center for k in ['center_x', 'center_y', 'radius']):
                raise HTTPException(status_code=400, detail="中心點格式錯誤")
            
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
    
    def crop_seals(self, image_id: UUID, seals: List[Dict], margin: int = 10) -> List[UUID]:
        """
        裁切圖像中的多個印鑑區域並保存為獨立圖像
        
        Args:
            image_id: 原圖像 ID
            seals: 印鑑列表，每個元素包含 bbox, center, confidence
            margin: 邊距（像素），默認10
            
        Returns:
            裁切後的圖像 ID 列表
        """
        db_image = self.get_image(image_id)
        if not db_image:
            raise HTTPException(status_code=404, detail="圖像不存在")
        
        file_path = Path(db_image.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="圖像文件不存在")
        
        # 讀取原圖
        image = cv2.imread(str(file_path))
        if image is None:
            raise HTTPException(status_code=500, detail="無法讀取圖像文件")
        
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
        is_image2: bool = False
    ) -> Tuple[np.ndarray, Optional[float], Optional[Tuple[int, int]], Optional[float], Optional[Dict[str, float]]]:
        """
        去背景並對齊圖像（與 comparison_service 保持一致）
        
        Args:
            image: 待處理圖像（numpy array）
            reference_image: 參考圖像（用於圖像2的對齊優化，應該是已去背景的圖像）
            is_image2: 是否為圖像2（需要相對於圖像1優化）
            
        Returns:
            如果 is_image2=True 且 reference_image 不為 None:
                (對齊後的圖像, 旋轉角度, (x偏移, y偏移), 相似度, 詳細指標)
            否則:
                (去背景後的圖像, None, None, None, None)
        """
        comparator = SealComparator()
        
        # 去背景
        try:
            image = comparator._auto_detect_bounds_and_remove_background(image)
        except Exception as e:
            print(f"警告：自動外框偵測失敗，使用原圖: {str(e)}")
            # 如果處理失敗，使用原圖繼續處理
            pass
        
        # 對齊處理
        if is_image2 and reference_image is not None:
            # 圖像2：相對於圖像1進行優化對齊
            try:
                # reference_image 已經是去背景後的圖像（img1_no_bg），不需要再次處理
                image_aligned, angle, offset, similarity, metrics = comparator._align_image2_to_image1(
                    reference_image, image, rotation_range=45.0, translation_range=100
                )
                return image_aligned, angle, offset, similarity, metrics
            except Exception as e:
                print(f"警告：圖像2對齊優化失敗，使用原圖: {str(e)}")
                # 如果對齊失敗，直接返回原圖
                return image, None, None, None, None
        else:
            # 圖像1：只去背景，不對齊
            return image, None, None, None, None
    
    def compare_image1_with_seals(
        self, 
        image1_id: UUID, 
        seal_image_ids: List[UUID],
        threshold: float = 0.95
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
        # 獲取圖像1
        image1 = self.get_image(image1_id)
        if not image1:
            raise HTTPException(status_code=404, detail="圖像1不存在")
        
        image1_path = Path(image1.file_path)
        if not image1_path.exists():
            raise HTTPException(status_code=404, detail="圖像1文件不存在")
        
        # 檢查圖像1是否有標記印鑑位置
        if not image1.seal_bbox:
            raise HTTPException(status_code=400, detail="圖像1未標記印鑑位置，無法裁切")
        
        # 讀取原始圖像1
        image1_original = cv2.imread(str(image1_path))
        if image1_original is None:
            raise HTTPException(status_code=500, detail="無法讀取圖像1文件")
        
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
            raise HTTPException(status_code=400, detail="裁切區域太小")
        
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
        
        results = []
        comparator = SealComparator(threshold=threshold)
        
        # 對每個印鑑進行比對
        for idx, seal_image_id in enumerate(seal_image_ids):
            result = {
                'seal_index': idx + 1,
                'seal_image_id': seal_image_id,
                'similarity': None,
                'is_match': None,
                'overlay1_path': None,
                'overlay2_path': None,
                'heatmap_path': None,
                'alignment_angle': None,
                'alignment_offset': None,
                'alignment_similarity': None,
                'alignment_success': False,
                'error': None
            }
            
            try:
                # 獲取印鑑圖像
                seal_image = self.get_image(seal_image_id)
                if not seal_image:
                    result['error'] = "印鑑圖像不存在"
                    results.append(result)
                    continue
                
                seal_image_path = Path(seal_image.file_path)
                if not seal_image_path.exists():
                    result['error'] = "印鑑圖像文件不存在"
                    results.append(result)
                    continue
                
                # 執行比對
                # 使用裁切後的圖像1和圖像2進行比對
                # 流程：載入圖像 → 去背景 → 旋轉對齊 → 保存文件 → 計算相似度
                
                # 生成記錄 ID（用於文件命名）
                record_id = f"{image1_id}_{seal_image_id}_{idx+1}"
                
                # 1. 載入圖像
                img1_original = cv2.imread(str(image1_cropped_path))
                img2_original = cv2.imread(str(seal_image_path))
                
                if img1_original is None:
                    result['error'] = "無法載入圖像1"
                    results.append(result)
                    continue
                
                if img2_original is None:
                    result['error'] = "無法載入印鑑圖像"
                    results.append(result)
                    continue
                
                # 2. 去背景處理（圖像1）
                img1_no_bg, _, _, _, _ = self._remove_background_and_align(img1_original.copy())
                
                # 3. 去背景和對齊處理（圖像2，相對於圖像1）
                img2_aligned, alignment_angle, alignment_offset, alignment_similarity, alignment_metrics = self._remove_background_and_align(
                    img2_original.copy(),
                    reference_image=img1_no_bg,  # 傳入去背景後的圖像1作為參考
                    is_image2=True
                )
                
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
                    similarity_str = f"{alignment_similarity:.4f}" if alignment_similarity is not None else 'N/A'
                    print(f"印鑑 {idx+1} 對齊成功: 角度={alignment_angle:.2f}度, 偏移=({alignment_offset[0]}, {alignment_offset[1]}), 相似度={similarity_str}")
                else:
                    result['alignment_success'] = False
                    print(f"警告：印鑑 {idx+1} 對齊失敗，使用去背景後的圖像（未對齊）")
                
                # 4. 保存對齊後的圖像到文件（在比對之前）
                image1_for_comparison = comparison_dir / f"image1_cropped_{record_id}.jpg"
                cv2.imwrite(str(image1_for_comparison), img1_no_bg)
                
                image2_for_comparison = comparison_dir / f"image2_cropped_{record_id}.jpg"
                cv2.imwrite(str(image2_for_comparison), img2_aligned)
                
                # 5. 使用 compare_files 比對（傳入文件路徑）
                try:
                    is_match, similarity, details, img2_corrected, img1_corrected = comparator.compare_files(
                        str(image1_for_comparison),
                        str(image2_for_comparison),
                        enable_rotation_search=False,  # 已經對齊，不需要旋轉搜索
                        enable_translation_search=False,  # 已經對齊，不需要平移搜索
                        bbox1=None,  # 已經裁切並對齊好了
                        bbox2=None   # 已經裁切並對齊好了
                    )
                    
                    result['similarity'] = float(similarity)
                    result['is_match'] = is_match
                except Exception as e:
                    result['error'] = f"相似度計算失敗: {str(e)}"
                    results.append(result)
                    continue
                
                # 6. 保存校正後的圖像（使用 compare_files 返回的圖像）
                image1_corrected_path = None
                image2_corrected_path = None
                
                if img1_corrected is not None:
                    corrected_file1 = comparison_dir / f"corrected_image1_{record_id}.jpg"
                    cv2.imwrite(str(corrected_file1), img1_corrected)
                    image1_corrected_path = str(corrected_file1)
                else:
                    # 如果 compare_files 沒有返回 img1_corrected，使用已保存的去背景圖像
                    image1_corrected_path = str(image1_for_comparison)
                
                if img2_corrected is not None:
                    corrected_file2 = comparison_dir / f"corrected_{record_id}.jpg"
                    cv2.imwrite(str(corrected_file2), img2_corrected)
                    image2_corrected_path = str(corrected_file2)
                else:
                    # 如果 compare_files 沒有返回 img2_corrected，使用已保存的對齊圖像
                    image2_corrected_path = str(image2_for_comparison)
                
                # 生成疊圖（使用對齊後的圖像）
                try:
                    # 使用對齊後的圖像1和圖像2生成疊圖
                    # 優先使用 compare_files 返回的校正圖像，否則使用已保存的對齊圖像
                    image1_for_overlay = image1_corrected_path if image1_corrected_path else str(image1_for_comparison)
                    image2_for_overlay = image2_corrected_path if image2_corrected_path else str(image2_for_comparison)
                    
                    overlay1_path, overlay2_path = create_overlay_image(
                        image1_for_overlay,
                        str(seal_image_path),
                        comparison_dir,
                        record_id,
                        image2_corrected_path=image2_for_overlay  # 使用對齊後的圖像2
                    )
                    if overlay1_path:
                        # 只保存文件名，前端通過 API 獲取
                        result['overlay1_path'] = Path(overlay1_path).name
                    if overlay2_path:
                        result['overlay2_path'] = Path(overlay2_path).name
                except Exception as e:
                    print(f"生成疊圖失敗 (印鑑 {idx+1}): {e}")
                    # 不設置錯誤，繼續處理
                
                # 生成熱力圖（使用對齊後的圖像）
                try:
                    # create_difference_heatmap 需要 record_id 為 int，但我們使用字符串
                    # 使用 hash 轉換為數字
                    record_id_int = hash(record_id) % (10 ** 9)  # 轉換為正整數
                    
                    # 優先使用 compare_files 返回的校正圖像，否則使用已保存的對齊圖像
                    image1_for_heatmap = image1_corrected_path if image1_corrected_path else str(image1_for_comparison)
                    image2_for_heatmap = image2_corrected_path if image2_corrected_path else str(image2_for_comparison)
                    
                    heatmap_path, _ = create_difference_heatmap(
                        image1_for_heatmap,  # 使用對齊後的圖像1
                        image2_for_heatmap,  # 使用對齊後的圖像2
                        str(seal_image_path),
                        comparison_dir,
                        record_id_int
                    )
                    if heatmap_path:
                        # create_difference_heatmap 返回相對路徑 "heatmaps/heatmap_{record_id}.jpg"
                        # 需要轉換為絕對路徑再提取文件名
                        if not Path(heatmap_path).is_absolute():
                            heatmap_path = comparison_dir / heatmap_path
                        result['heatmap_path'] = Path(heatmap_path).name
                except Exception as e:
                    print(f"生成熱力圖失敗 (印鑑 {idx+1}): {e}")
                    # 不設置錯誤，繼續處理
                
            except Exception as e:
                result['error'] = str(e)
                print(f"比對印鑑 {idx+1} 時出錯: {e}")
            
            results.append(result)
        
        return results
    
    def detect_matching_seals_with_rotation(
        self,
        image1_id: UUID,
        image2_id: UUID,
        rotation_range: float = 45.0,
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
            raise HTTPException(status_code=404, detail="圖像1不存在")
        
        image1_path = Path(image1.file_path)
        if not image1_path.exists():
            raise HTTPException(status_code=404, detail="圖像1文件不存在")
        
        # 獲取圖像2
        image2 = self.get_image(image2_id)
        if not image2:
            raise HTTPException(status_code=404, detail="圖像2不存在")
        
        image2_path = Path(image2.file_path)
        if not image2_path.exists():
            raise HTTPException(status_code=404, detail="圖像2文件不存在")
        
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

