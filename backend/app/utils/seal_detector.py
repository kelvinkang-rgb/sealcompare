"""
印鑑位置檢測模組（優化版）
用於快速檢測圖像中印鑑的位置（邊界框和中心點）
添加超時機制和性能優化，避免阻塞
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import signal
import time


class TimeoutError(Exception):
    """超時異常"""
    pass


def _timeout_handler(signum, frame):
    """超時處理函數"""
    raise TimeoutError("檢測超時")


def detect_seal_location(image_path: str, timeout: float = 3.0) -> Dict:
    """
    檢測圖像中印鑑的位置（帶超時保護）
    
    Args:
        image_path: 圖像文件路徑
        timeout: 超時時間（秒），默認3秒
        
    Returns:
        包含檢測結果的字典：
        {
            'detected': bool,
            'confidence': float,
            'bbox': {'x': int, 'y': int, 'width': int, 'height': int} or None,
            'center': {'center_x': int, 'center_y': int, 'radius': float} or None
        }
    """
    start_time = time.time()
    
    try:
        # 讀取圖像
        if not Path(image_path).exists():
            return _create_failed_result("圖像文件不存在")
        
        image = cv2.imread(image_path)
        if image is None:
            return _create_failed_result("無法讀取圖像文件")
        
        if image.size == 0:
            return _create_failed_result("圖像為空")
        
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return _create_failed_result("圖像尺寸無效")
        
        # 檢查是否已超時
        if time.time() - start_time > timeout:
            return _create_failed_result("檢測超時")
        
        # 使用快速檢測方法（優先使用輪廓檢測，最快）
        result = _detect_seal_fast(image, timeout - (time.time() - start_time))
        
        return result
        
    except TimeoutError:
        return _create_failed_result("檢測超時")
    except Exception as e:
        return _create_failed_result(f"檢測過程出錯: {str(e)}")


def _detect_seal_fast(image: np.ndarray, remaining_time: float) -> Dict:
    """
    快速檢測印鑑（優先使用輪廓檢測）
    
    Args:
        image: 圖像數組
        remaining_time: 剩餘時間（秒）
        
    Returns:
        檢測結果字典
    """
    start_time = time.time()
    h, w = image.shape[:2]
    
    # 方法1: 快速輪廓檢測（最快）
    if remaining_time > 1.0:
        result1 = _detect_by_contours_fast(image)
        if result1['detected'] and result1['confidence'] > 0.6:
            return result1
    
    # 方法2: 自適應閾值（較快）
    if remaining_time > 0.5:
        result2 = _detect_by_adaptive_threshold(image)
        if result2['detected']:
            return result2
    
    # 如果都失敗，返回第一個結果或失敗結果
    return result1 if 'result1' in locals() and result1['detected'] else _create_failed_result("未檢測到印鑑")


def _detect_by_contours_fast(image: np.ndarray) -> Dict:
    """快速輪廓檢測方法"""
    try:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 使用 Canny 邊緣檢測
        edges = cv2.Canny(gray, 50, 150)
        
        # 簡單的形態學操作（減少迭代次數）
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 查找輪廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return _create_failed_result("未找到輪廓")
        
        # 篩選輪廓（基於面積）
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < (w * h * 0.01):  # 太小
                continue
            if area > (w * h * 0.9):  # 太大
                continue
            circularity = _calculate_circularity(contour)
            if circularity > 0.2:  # 降低圓度要求以加快檢測
                valid_contours.append((contour, area, circularity))
        
        if not valid_contours:
            return _create_failed_result("未找到有效的輪廓")
        
        # 選擇面積最大的輪廓
        best_contour = max(valid_contours, key=lambda x: x[1])[0]
        
        # 計算邊界框和中心點
        bbox, center, radius = _calculate_bbox_and_center(best_contour)
        
        # 計算置信度
        area = cv2.contourArea(best_contour)
        circularity = _calculate_circularity(best_contour)
        area_ratio = area / (w * h)
        confidence = min(0.9, 0.4 + circularity * 0.3 + min(area_ratio * 5, 0.2))
        
        return {
            'detected': True,
            'confidence': confidence,
            'bbox': bbox,
            'center': center
        }
        
    except Exception as e:
        return _create_failed_result(f"輪廓檢測失敗: {str(e)}")


def _detect_by_adaptive_threshold(image: np.ndarray) -> Dict:
    """基於自適應閾值的檢測方法"""
    try:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 使用自適應閾值
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 簡單的形態學操作
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 查找輪廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return _create_failed_result("未找到輪廓")
        
        # 找到最大的輪廓
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < (w * h * 0.01):
            return _create_failed_result("檢測區域太小")
        
        # 計算邊界框和中心點
        bbox, center, radius = _calculate_bbox_and_center(largest_contour)
        
        # 計算置信度
        circularity = _calculate_circularity(largest_contour)
        area_ratio = area / (w * h)
        confidence = min(0.8, 0.3 + circularity * 0.3 + min(area_ratio * 5, 0.2))
        
        return {
            'detected': True,
            'confidence': confidence,
            'bbox': bbox,
            'center': center
        }
        
    except Exception as e:
        return _create_failed_result(f"自適應閾值檢測失敗: {str(e)}")


def _calculate_bbox_and_center(contour: np.ndarray) -> Tuple[Dict, Dict, float]:
    """
    計算輪廓的邊界框和中心點
    
    Returns:
        (bbox_dict, center_dict, radius)
    """
    # 邊界框
    x, y, w, h = cv2.boundingRect(contour)
    bbox = {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
    
    # 中心點和半徑
    M = cv2.moments(contour)
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
    else:
        # 如果無法計算質心，使用邊界框中心
        center_x = x + w // 2
        center_y = y + h // 2
    
    # 計算半徑（使用邊界框的對角線長度的一半作為近似）
    radius = np.sqrt(w**2 + h**2) / 2.0
    
    center = {
        'center_x': int(center_x),
        'center_y': int(center_y),
        'radius': float(radius)
    }
    
    return bbox, center, radius


def _calculate_circularity(contour: np.ndarray) -> float:
    """計算輪廓的圓度（0-1，1表示完美圓形）"""
    area = cv2.contourArea(contour)
    if area == 0:
        return 0.0
    
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    
    # 圓度 = 4π * 面積 / 周長²
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return min(1.0, circularity)


def _validate_detection(result: Dict, image_width: int, image_height: int) -> bool:
    """驗證檢測結果的合理性"""
    if not result['detected']:
        return False
    
    bbox = result.get('bbox')
    center = result.get('center')
    
    if not bbox or not center:
        return False
    
    # 檢查邊界框是否在圖像範圍內
    if (bbox['x'] < 0 or bbox['y'] < 0 or 
        bbox['x'] + bbox['width'] > image_width or
        bbox['y'] + bbox['height'] > image_height):
        return False
    
    # 檢查中心點是否在圖像範圍內
    if (center['center_x'] < 0 or center['center_x'] > image_width or
        center['center_y'] < 0 or center['center_y'] > image_height):
        return False
    
    # 檢查尺寸是否合理
    if bbox['width'] < 10 or bbox['height'] < 10:
        return False
    
    if bbox['width'] > image_width * 0.95 or bbox['height'] > image_height * 0.95:
        return False
    
    return True


def _create_failed_result(reason: str = "") -> Dict:
    """創建失敗的檢測結果"""
    return {
        'detected': False,
        'confidence': 0.0,
        'bbox': None,
        'center': None,
        'reason': reason
    }

