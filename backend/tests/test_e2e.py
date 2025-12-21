"""
端到端測試
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys
import tempfile
import shutil

# 這些測試標記為 e2e，可能需要較長時間運行
pytestmark = pytest.mark.e2e


class TestE2E:
    """端到端測試類"""
    
    @pytest.fixture
    def test_images(self, temp_dir):
        """創建測試圖像"""
        img1 = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.circle(img1, (100, 100), 80, (0, 0, 255), -1)
        img1_path = temp_dir / "seal1.jpg"
        cv2.imwrite(str(img1_path), img1)
        
        img2 = img1.copy()
        img2_path = temp_dir / "seal2.jpg"
        cv2.imwrite(str(img2_path), img2)
        
        return str(img1_path), str(img2_path)
    
    def test_full_comparison_flow(self, test_images):
        """測試完整比對流程"""
        from seal_compare import SealComparator
        
        img1_path, img2_path = test_images
        
        comparator = SealComparator(threshold=0.95)
        is_match, similarity, details, img2_corrected = comparator.compare_files(
            img1_path,
            img2_path,
            enable_rotation_search=True
        )
        
        # 驗證結果
        assert is_match == True
        assert similarity >= 0.95
        assert 'ssim' in details
        assert 'template_match' in details
        assert 'pixel_diff' in details
        assert 'rotation_angle' in details

