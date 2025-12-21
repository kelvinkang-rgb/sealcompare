"""
Overlay 模組單元測試
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys
import tempfile
import shutil

# 添加 core 目錄到路徑
core_path = Path(__file__).parent.parent / "core"
sys.path.insert(0, str(core_path))

from overlay import create_overlay_image


class TestOverlay:
    """Overlay 模組測試類"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """創建臨時輸出目錄"""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def test_images(self, temp_dir):
        """創建測試圖像文件"""
        img1 = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.circle(img1, (100, 100), 80, (0, 0, 255), -1)
        img1_path = temp_dir / "img1.jpg"
        cv2.imwrite(str(img1_path), img1)
        
        img2 = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.circle(img2, (100, 100), 75, (0, 0, 255), -1)  # 稍小的圓
        img2_path = temp_dir / "img2.jpg"
        cv2.imwrite(str(img2_path), img2)
        
        return str(img1_path), str(img2_path)
    
    def test_create_overlay_image(self, test_images, temp_output_dir):
        """測試疊圖生成"""
        img1_path, img2_path = test_images
        
        overlay1, overlay2 = create_overlay_image(
            img1_path,
            img2_path,
            temp_output_dir,
            "test_001",
            image2_corrected_path=None
        )
        
        assert overlay1 is not None
        assert overlay2 is not None
        assert Path(overlay1).exists()
        assert Path(overlay2).exists()
    
    def test_create_overlay_image_with_corrected(self, test_images, temp_output_dir):
        """測試疊圖生成（帶校正圖像）"""
        img1_path, img2_path = test_images
        
        overlay1, overlay2 = create_overlay_image(
            img1_path,
            img2_path,
            temp_output_dir,
            "test_002",
            image2_corrected_path=img2_path
        )
        
        assert overlay1 is not None
        assert overlay2 is not None

