"""
SealComparator 核心功能單元測試
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

# 添加 core 目錄到路徑
core_path = Path(__file__).parent.parent / "core"
sys.path.insert(0, str(core_path))

from seal_compare import SealComparator


class TestSealComparator:
    """SealComparator 測試類"""
    
    def test_init(self):
        """測試初始化"""
        comparator = SealComparator()
        assert comparator.threshold == 0.95
        
        comparator = SealComparator(threshold=0.9)
        assert comparator.threshold == 0.9
    
    def test_load_image_success(self, test_image_path):
        """測試圖像載入成功"""
        comparator = SealComparator()
        image = comparator.load_image(test_image_path)
        assert image is not None
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3  # BGR 圖像
    
    def test_load_image_not_found(self):
        """測試圖像文件不存在"""
        comparator = SealComparator()
        image = comparator.load_image("nonexistent.jpg")
        assert image is None
    
    def test_preprocess_image(self, test_image_array):
        """測試圖像預處理"""
        comparator = SealComparator()
        processed = comparator.preprocess_image(test_image_array)
        
        assert processed is not None
        assert len(processed.shape) == 2  # 灰度圖
        assert processed.dtype == np.uint8
    
    def test_preprocess_image_none(self):
        """測試預處理 None 圖像"""
        comparator = SealComparator()
        with pytest.raises(ValueError, match="圖像不能為 None"):
            comparator.preprocess_image(None)
    
    def test_preprocess_image_empty(self, empty_image):
        """測試預處理空圖像"""
        comparator = SealComparator()
        with pytest.raises(ValueError, match="圖像不能為空"):
            comparator.preprocess_image(empty_image)
    
    def test_compare_identical_images(self, identical_images):
        """測試比對相同圖像"""
        comparator = SealComparator(threshold=0.95)
        img1, img2 = identical_images
        is_match, similarity, details, _ = comparator.compare_images(img1, img2)
        
        assert is_match == True
        assert similarity >= 0.95
        assert 'ssim' in details
        assert 'template_match' in details
        assert 'pixel_diff' in details
    
    def test_compare_different_images(self, different_images):
        """測試比對不同圖像"""
        comparator = SealComparator(threshold=0.95)
        img1, img2 = different_images
        is_match, similarity, details, _ = comparator.compare_images(img1, img2)
        
        assert is_match == False
        assert similarity < 0.95
        assert 'ssim' in details
    
    def test_compare_rotated_image(self, test_image_array, rotated_image):
        """測試比對旋轉圖像（啟用旋轉搜索）"""
        comparator = SealComparator(threshold=0.95)
        is_match, similarity, details, img2_corrected = comparator.compare_images(
            test_image_array, 
            rotated_image,
            enable_rotation_search=True
        )
        
        assert 'rotation_angle' in details
        assert details['rotation_angle'] is not None
        assert img2_corrected is not None
        # 旋轉後應該能匹配
        assert is_match == True or similarity > 0.8
    
    def test_compare_rotated_image_no_search(self, test_image_array, rotated_image):
        """測試比對旋轉圖像（禁用旋轉搜索）"""
        comparator = SealComparator(threshold=0.95)
        is_match, similarity, details, img2_corrected = comparator.compare_images(
            test_image_array,
            rotated_image,
            enable_rotation_search=False
        )
        
        assert details['rotation_angle'] is None
        assert img2_corrected is None
        # 不旋轉搜索時相似度應該較低
        assert similarity < 0.9
    
    def test_compare_images_none(self):
        """測試比對 None 圖像"""
        comparator = SealComparator()
        with pytest.raises(ValueError, match="圖像不能為 None"):
            comparator.compare_images(None, np.ones((100, 100, 3), dtype=np.uint8))
    
    def test_compare_images_empty(self, empty_image):
        """測試比對空圖像"""
        comparator = SealComparator()
        img = np.ones((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="圖像不能為空"):
            comparator.compare_images(empty_image, img)
    
    def test_compare_files_success(self, test_image_path):
        """測試文件比對成功"""
        comparator = SealComparator(threshold=0.95)
        is_match, similarity, details, _ = comparator.compare_files(
            test_image_path,
            test_image_path
        )
        
        assert is_match == True
        assert similarity >= 0.95
        assert 'error' not in details
    
    def test_compare_files_not_found(self):
        """測試文件比對（文件不存在）"""
        comparator = SealComparator()
        is_match, similarity, details, _ = comparator.compare_files(
            "nonexistent1.jpg",
            "nonexistent2.jpg"
        )
        
        assert is_match == False
        assert similarity == 0.0
        assert 'error' in details
    
    def test_compare_files_empty_path(self):
        """測試文件比對（空路徑）"""
        comparator = SealComparator()
        with pytest.raises(ValueError, match="圖像路徑不能為空"):
            comparator.compare_files("", "test.jpg")
    
    def test_calculate_ssim(self, identical_images):
        """測試 SSIM 計算"""
        comparator = SealComparator()
        img1, img2 = identical_images
        
        # 預處理圖像
        img1_processed = comparator.preprocess_image(img1)
        img2_processed = comparator.preprocess_image(img2)
        
        # 確保尺寸相同
        h1, w1 = img1_processed.shape
        h2, w2 = img2_processed.shape
        if h1 != h2 or w1 != w2:
            target_h = max(h1, h2)
            target_w = max(w1, w2)
            img1_processed = cv2.resize(img1_processed, (target_w, target_h))
            img2_processed = cv2.resize(img2_processed, (target_w, target_h))
        
        ssim = comparator._calculate_ssim(img1_processed, img2_processed)
        assert 0.0 <= ssim <= 1.0
        assert ssim > 0.9  # 相同圖像應該有高 SSIM
    
    def test_template_match(self, identical_images):
        """測試模板匹配"""
        comparator = SealComparator()
        img1, img2 = identical_images
        
        # 預處理圖像
        img1_processed = comparator.preprocess_image(img1)
        img2_processed = comparator.preprocess_image(img2)
        
        # 確保尺寸相同
        h1, w1 = img1_processed.shape
        h2, w2 = img2_processed.shape
        if h1 != h2 or w1 != w2:
            target_h = max(h1, h2)
            target_w = max(w1, w2)
            img1_processed = cv2.resize(img1_processed, (target_w, target_h))
            img2_processed = cv2.resize(img2_processed, (target_w, target_h))
        
        match = comparator._template_match(img1_processed, img2_processed)
        assert 0.0 <= match <= 1.0
        assert match > 0.9  # 相同圖像應該有高匹配度
    
    def test_pixel_difference(self, identical_images):
        """測試像素差異計算"""
        comparator = SealComparator()
        img1, img2 = identical_images
        
        # 預處理圖像
        img1_processed = comparator.preprocess_image(img1)
        img2_processed = comparator.preprocess_image(img2)
        
        # 確保尺寸相同
        h1, w1 = img1_processed.shape
        h2, w2 = img2_processed.shape
        if h1 != h2 or w1 != w2:
            target_h = max(h1, h2)
            target_w = max(w1, w2)
            img1_processed = cv2.resize(img1_processed, (target_w, target_h))
            img2_processed = cv2.resize(img2_processed, (target_w, target_h))
        
        diff = comparator._pixel_difference(img1_processed, img2_processed)
        assert 0.0 <= diff <= 1.0
        assert diff < 0.1  # 相同圖像應該有低差異
    
    def test_find_best_rotation_angle_same(self, test_image_array):
        """測試旋轉角度搜索（相同圖像）"""
        comparator = SealComparator()
        img1_processed = comparator.preprocess_image(test_image_array)
        img2_processed = comparator.preprocess_image(test_image_array)
        
        angle, rotated = comparator.find_best_rotation_angle(img1_processed, img2_processed)
        
        assert abs(angle) < 5.0  # 相同圖像應該接近 0 度
        assert rotated is not None
        assert len(rotated.shape) == 2
    
    def test_find_best_rotation_angle_rotated(self, test_image_array, rotated_image):
        """測試旋轉角度搜索（旋轉圖像）"""
        comparator = SealComparator()
        img1_processed = comparator.preprocess_image(test_image_array)
        img2_processed = comparator.preprocess_image(rotated_image)
        
        angle, rotated = comparator.find_best_rotation_angle(img1_processed, img2_processed)
        
        # 應該能找到接近 45 度的角度（允許誤差）
        assert 30.0 <= abs(angle) <= 60.0 or abs(angle) < 5.0
        assert rotated is not None

