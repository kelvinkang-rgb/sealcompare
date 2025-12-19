"""
ComparisonService 服務層單元測試
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys
import tempfile
import shutil
from uuid import uuid4
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# 添加 app 目錄到路徑
app_path = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_path))

from app.database import Base
from app.models import Image, Comparison, ComparisonStatus
from app.services.comparison_service import ComparisonService
from app.schemas import ComparisonCreate


@pytest.fixture
def test_db():
    """創建測試數據庫"""
    # 使用內存 SQLite 數據庫進行測試
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def test_images(test_db, temp_dir):
    """創建測試圖像記錄"""
    # 創建測試圖像文件
    img1 = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.circle(img1, (100, 100), 80, (0, 0, 255), -1)
    img1_path = temp_dir / "img1.jpg"
    cv2.imwrite(str(img1_path), img1)
    
    img2 = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.circle(img2, (100, 100), 75, (0, 0, 255), -1)
    img2_path = temp_dir / "img2.jpg"
    cv2.imwrite(str(img2_path), img2)
    
    # 創建數據庫記錄
    image1 = Image(
        id=uuid4(),
        filename="img1.jpg",
        file_path=str(img1_path),
        file_size="100KB",
        mime_type="image/jpeg"
    )
    image2 = Image(
        id=uuid4(),
        filename="img2.jpg",
        file_path=str(img2_path),
        file_size="100KB",
        mime_type="image/jpeg"
    )
    
    test_db.add(image1)
    test_db.add(image2)
    test_db.commit()
    test_db.refresh(image1)
    test_db.refresh(image2)
    
    return image1, image2


class TestComparisonService:
    """ComparisonService 測試類"""
    
    def test_create_comparison(self, test_db, test_images):
        """測試創建比對記錄"""
        service = ComparisonService(test_db)
        image1, image2 = test_images
        
        comparison_data = ComparisonCreate(
            image1_id=image1.id,
            image2_id=image2.id,
            threshold=0.95
        )
        
        comparison = service.create_comparison(comparison_data)
        
        assert comparison is not None
        assert comparison.image1_id == image1.id
        assert comparison.image2_id == image2.id
        assert comparison.threshold == 0.95
        assert comparison.status == ComparisonStatus.PENDING
    
    def test_create_comparison_invalid_images(self, test_db):
        """測試創建比對記錄（無效圖像）"""
        service = ComparisonService(test_db)
        
        comparison_data = ComparisonCreate(
            image1_id=uuid4(),
            image2_id=uuid4(),
            threshold=0.95
        )
        
        with pytest.raises(ValueError, match="圖像不存在"):
            service.create_comparison(comparison_data)
    
    def test_update_progress(self, test_db, test_images):
        """測試進度更新"""
        service = ComparisonService(test_db)
        image1, image2 = test_images
        
        # 創建比對記錄
        comparison_data = ComparisonCreate(
            image1_id=image1.id,
            image2_id=image2.id,
            threshold=0.95
        )
        comparison = service.create_comparison(comparison_data)
        
        # 更新進度
        service._update_progress(comparison.id, 50.0, "處理中", "測試步驟")
        
        # 驗證更新
        test_db.refresh(comparison)
        assert comparison.progress == 50.0
        assert comparison.progress_message == "處理中"
        assert comparison.current_step == "測試步驟"
    
    def test_get_comparison(self, test_db, test_images):
        """測試獲取比對記錄"""
        service = ComparisonService(test_db)
        image1, image2 = test_images
        
        # 創建比對記錄
        comparison_data = ComparisonCreate(
            image1_id=image1.id,
            image2_id=image2.id,
            threshold=0.95
        )
        created = service.create_comparison(comparison_data)
        
        # 獲取比對記錄
        retrieved = service.get_comparison(created.id)
        
        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.image1_id == image1.id
    
    def test_list_comparisons(self, test_db, test_images):
        """測試列出比對記錄"""
        service = ComparisonService(test_db)
        image1, image2 = test_images
        
        # 創建多個比對記錄
        for i in range(3):
            comparison_data = ComparisonCreate(
                image1_id=image1.id,
                image2_id=image2.id,
                threshold=0.95
            )
            service.create_comparison(comparison_data)
        
        # 列出比對記錄
        comparisons = service.list_comparisons(skip=0, limit=10)
        
        assert len(comparisons) == 3
    
    def test_update_comparison(self, test_db, test_images):
        """測試更新比對記錄"""
        service = ComparisonService(test_db)
        image1, image2 = test_images
        
        # 創建比對記錄
        comparison_data = ComparisonCreate(
            image1_id=image1.id,
            image2_id=image2.id,
            threshold=0.95
        )
        comparison = service.create_comparison(comparison_data)
        
        # 更新比對記錄
        update_data = {
            'threshold': 0.9,
            'notes': '測試備註'
        }
        updated = service.update_comparison(comparison.id, update_data)
        
        assert updated.threshold == 0.9
        assert updated.notes == '測試備註'
    
    def test_delete_comparison(self, test_db, test_images):
        """測試軟刪除比對記錄"""
        service = ComparisonService(test_db)
        image1, image2 = test_images
        
        # 創建比對記錄
        comparison_data = ComparisonCreate(
            image1_id=image1.id,
            image2_id=image2.id,
            threshold=0.95
        )
        comparison = service.create_comparison(comparison_data)
        
        # 刪除比對記錄
        result = service.delete_comparison(comparison.id)
        assert result == True
        
        # 驗證軟刪除
        test_db.refresh(comparison)
        assert comparison.deleted_at is not None
        
        # 驗證查詢時不包含已刪除記錄
        retrieved = service.get_comparison(comparison.id)
        assert retrieved is None

