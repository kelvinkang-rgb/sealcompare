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
from app.utils.seal_detector import detect_seal_location
from fastapi import UploadFile, HTTPException
from app.config import settings
from typing import Dict, Optional


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

