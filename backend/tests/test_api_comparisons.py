"""
API 端點集成測試
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import sys
from pathlib import Path

# 添加 app 目錄到路徑
app_path = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_path))

from app.database import Base, get_db
from app.main import app
from app.models import Image, Comparison
import numpy as np
import cv2
from uuid import uuid4
import tempfile


@pytest.fixture
def test_db():
    """創建測試數據庫"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    
    app.dependency_overrides[get_db] = override_get_db
    yield TestingSessionLocal()
    app.dependency_overrides.clear()
    Base.metadata.drop_all(engine)


@pytest.fixture
def client(test_db):
    """創建測試客戶端"""
    return TestClient(app)


@pytest.fixture
def test_image_file(temp_dir):
    """創建測試圖像文件"""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.circle(img, (100, 100), 80, (0, 0, 255), -1)
    img_path = temp_dir / "test.jpg"
    cv2.imwrite(str(img_path), img)
    return str(img_path)


class TestAPIComparisons:
    """API 端點測試類"""
    
    def test_create_comparison(self, client, test_db, test_image_file):
        """測試創建比對 API"""
        # 先上傳圖像
        with open(test_image_file, "rb") as f:
            response1 = client.post(
                "/api/v1/images/upload",
                files={"file": ("test1.jpg", f, "image/jpeg")}
            )
        assert response1.status_code == 201
        image1_id = response1.json()["id"]
        
        with open(test_image_file, "rb") as f:
            response2 = client.post(
                "/api/v1/images/upload",
                files={"file": ("test2.jpg", f, "image/jpeg")}
            )
        assert response2.status_code == 201
        image2_id = response2.json()["id"]
        
        # 創建比對
        response = client.post(
            "/api/v1/comparisons/",
            json={
                "image1_id": image1_id,
                "image2_id": image2_id,
                "threshold": 0.95
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["image1_id"] == image1_id
        assert data["image2_id"] == image2_id
    
    def test_get_comparison(self, client, test_db, test_image_file):
        """測試獲取比對 API"""
        # 創建比對（簡化版，直接使用數據庫）
        db = next(test_db())
        img1 = Image(
            id=uuid4(),
            filename="test1.jpg",
            file_path=test_image_file,
            file_size="100KB",
            mime_type="image/jpeg"
        )
        img2 = Image(
            id=uuid4(),
            filename="test2.jpg",
            file_path=test_image_file,
            file_size="100KB",
            mime_type="image/jpeg"
        )
        db.add(img1)
        db.add(img2)
        db.commit()
        
        comparison = Comparison(
            image1_id=img1.id,
            image2_id=img2.id,
            threshold=0.95
        )
        db.add(comparison)
        db.commit()
        db.refresh(comparison)
        
        # 獲取比對
        response = client.get(f"/api/v1/comparisons/{comparison.id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(comparison.id)
    
    def test_get_comparison_status(self, client, test_db, test_image_file):
        """測試獲取比對狀態 API"""
        # 創建比對
        db = next(test_db())
        img1 = Image(
            id=uuid4(),
            filename="test1.jpg",
            file_path=test_image_file,
            file_size="100KB",
            mime_type="image/jpeg"
        )
        img2 = Image(
            id=uuid4(),
            filename="test2.jpg",
            file_path=test_image_file,
            file_size="100KB",
            mime_type="image/jpeg"
        )
        db.add(img1)
        db.add(img2)
        db.commit()
        
        comparison = Comparison(
            image1_id=img1.id,
            image2_id=img2.id,
            threshold=0.95
        )
        db.add(comparison)
        db.commit()
        db.refresh(comparison)
        
        # 獲取狀態
        response = client.get(f"/api/v1/comparisons/{comparison.id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "progress" in data

