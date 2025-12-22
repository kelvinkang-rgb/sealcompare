"""
Verification 模組單元測試
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

from verification import (
    create_correction_comparison,
    create_difference_heatmap,
    calculate_alignment_metrics
)


class TestVerification:
    """Verification 模組測試類"""
    
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
        
        img2 = img1.copy()
        img2_path = temp_dir / "img2.jpg"
        cv2.imwrite(str(img2_path), img2)
        
        return str(img1_path), str(img2_path)
    
    def test_create_correction_comparison(self, test_images, temp_output_dir):
        """測試並排對比圖生成（兩圖並排）"""
        img1_path, img2_path = test_images
        
        result = create_correction_comparison(
            img1_path,
            img2_path,  # 圖像2（已裁切、去背景和對齊）
            temp_output_dir,
            1,
            rotation_angle=0.0
        )
        
        assert result is not None
        assert Path(temp_output_dir / Path(result).name).exists()
    
    def test_create_correction_comparison_with_alignment(self, test_images, temp_output_dir):
        """測試並排對比圖生成（帶對齊參數）"""
        img1_path, img2_path = test_images
        
        result = create_correction_comparison(
            img1_path,
            img2_path,
            temp_output_dir,
            2,
            rotation_angle=5.5,
            translation_offset={'x': 10, 'y': -5}
        )
        
        assert result is not None
        assert Path(temp_output_dir / Path(result).name).exists()
    
    def test_create_difference_heatmap(self, test_images, temp_output_dir):
        """測試差異熱力圖生成"""
        img1_path, img2_path = test_images
        
        result, stats = create_difference_heatmap(
            img1_path,
            img2_path,  # 使用相同圖像
            img2_path,
            temp_output_dir,
            1
        )
        
        assert result is not None
        assert isinstance(stats, dict)
        assert 'diff_pixels' in stats
        assert 'diff_percentage' in stats
    
    def test_calculate_alignment_metrics(self):
        """測試對齊指標計算"""
        img1 = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.circle(img1, (100, 100), 80, (0, 0, 255), -1)
        
        img2 = img1.copy()
        
        metrics = calculate_alignment_metrics(
            img1,
            img2,
            img2,  # 使用相同圖像作為校正後圖像
            rotation_angle=0.0
        )
        
        assert isinstance(metrics, dict)
        assert 'center_offset' in metrics
        assert 'size_ratio' in metrics
        assert metrics['size_ratio'] > 0

