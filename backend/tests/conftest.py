"""
Pytest 配置和共享夾具
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """創建臨時目錄"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_image_path(temp_dir):
    """創建測試圖像文件"""
    # 創建一個簡單的測試圖像（紅色圓形印章）
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255  # 白色背景
    cv2.circle(img, (100, 100), 80, (0, 0, 255), -1)  # 紅色圓形
    img_path = temp_dir / "test_seal.jpg"
    cv2.imwrite(str(img_path), img)
    return str(img_path)


@pytest.fixture
def test_image_array():
    """創建測試圖像數組"""
    # 創建一個簡單的測試圖像（紅色圓形印章）
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255  # 白色背景
    cv2.circle(img, (100, 100), 80, (0, 0, 255), -1)  # 紅色圓形
    return img


@pytest.fixture
def identical_images():
    """創建兩個相同的圖像"""
    img1 = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.circle(img1, (100, 100), 80, (0, 0, 255), -1)
    img2 = img1.copy()
    return img1, img2


@pytest.fixture
def rotated_image(test_image_array):
    """創建旋轉後的圖像"""
    h, w = test_image_array.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(test_image_array, M, (w, h), borderValue=(255, 255, 255))
    return rotated


@pytest.fixture
def different_images():
    """創建兩個不同的圖像"""
    img1 = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.circle(img1, (100, 100), 80, (0, 0, 255), -1)  # 紅色圓形
    
    img2 = np.ones((200, 200, 3), dtype=np.uint8) * 255
    cv2.rectangle(img2, (50, 50), (150, 150), (0, 255, 0), -1)  # 綠色矩形
    
    return img1, img2


@pytest.fixture
def empty_image():
    """創建空圖像"""
    return np.array([])


@pytest.fixture
def small_image():
    """創建極小圖像"""
    return np.ones((10, 10, 3), dtype=np.uint8) * 255

