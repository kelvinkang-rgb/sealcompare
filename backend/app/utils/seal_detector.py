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
        if result1['detected'] and result1['confidence'] > 0.2:
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
    
    # 檢查尺寸是否合理（降低最小尺寸要求，放寬最大尺寸限制）
    if bbox['width'] < 5 or bbox['height'] < 5:
        return False
    
    if bbox['width'] > image_width * 0.98 or bbox['height'] > image_height * 0.98:
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


def detect_multiple_seals(image_path: str, timeout: float = 5.0, max_seals: int = 10) -> Dict:
    """
    檢測圖像中的多個印鑑位置（帶超時保護）
    
    Args:
        image_path: 圖像文件路徑
        timeout: 超時時間（秒），默認5秒
        max_seals: 最大檢測數量，默認10個
        
    Returns:
        包含檢測結果的字典：
        {
            'detected': bool,
            'seals': [
                {
                    'bbox': {'x': int, 'y': int, 'width': int, 'height': int},
                    'center': {'center_x': int, 'center_y': int, 'radius': float},
                    'confidence': float
                },
                ...
            ],
            'count': int
        }
    """
    start_time = time.time()
    
    try:
        # 讀取圖像
        if not Path(image_path).exists():
            return {
                'detected': False,
                'seals': [],
                'count': 0,
                'reason': '圖像文件不存在'
            }
        
        image = cv2.imread(image_path)
        if image is None:
            return {
                'detected': False,
                'seals': [],
                'count': 0,
                'reason': '無法讀取圖像文件'
            }
        
        if image.size == 0:
            return {
                'detected': False,
                'seals': [],
                'count': 0,
                'reason': '圖像為空'
            }
        
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return {
                'detected': False,
                'seals': [],
                'count': 0,
                'reason': '圖像尺寸無效'
            }
        
        # 檢查是否已超時
        if time.time() - start_time > timeout:
            return {
                'detected': False,
                'seals': [],
                'count': 0,
                'reason': '檢測超時'
            }
        
        # 使用多印鑑檢測方法
        result = _detect_multiple_seals_fast(image, timeout - (time.time() - start_time), max_seals)
        
        return result
        
    except TimeoutError:
        return {
            'detected': False,
            'seals': [],
            'count': 0,
            'reason': '檢測超時'
        }
    except Exception as e:
        return {
            'detected': False,
            'seals': [],
            'count': 0,
            'reason': f'檢測過程出錯: {str(e)}'
        }


def _detect_multiple_seals_fast(image: np.ndarray, remaining_time: float, max_seals: int = 10) -> Dict:
    """
    快速檢測多個印鑑（優先使用輪廓檢測）
    
    Args:
        image: 圖像數組
        remaining_time: 剩餘時間（秒）
        max_seals: 最大檢測數量
        
    Returns:
        檢測結果字典
    """
    start_time = time.time()
    h, w = image.shape[:2]
    
    # 使用輪廓檢測方法檢測所有符合條件的印鑑
    if remaining_time > 1.0:
        seals = _detect_multiple_by_contours(image, max_seals)
        if seals:
            return {
                'detected': True,
                'seals': seals,
                'count': len(seals)
            }
    
    # 如果沒有檢測到，返回空結果
    return {
        'detected': False,
        'seals': [],
        'count': 0,
        'reason': '未檢測到印鑑'
    }


def _detect_multiple_by_contours(image: np.ndarray, max_seals: int = 10) -> list:
    """使用輪廓檢測方法檢測多個印鑑"""
    try:
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 使用 Canny 邊緣檢測（降低閾值以檢測更多邊緣）
        edges = cv2.Canny(gray, 20, 80)
        
        # 簡單的形態學操作
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 查找輪廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # 篩選輪廓（基於面積和圓度）- 降低判定標準以檢測更多相似圖像
        valid_seals = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < (w * h * 0.001):  # 降低最小面積要求（從1%降到0.1%）
                continue
            if area > (w * h * 0.95):  # 放寬最大面積限制（從90%提高到95%）
                continue
            circularity = _calculate_circularity(contour)
            if circularity > 0.05:  # 降低圓度要求（從0.2降到0.05）
                # 計算邊界框和中心點
                bbox, center, radius = _calculate_bbox_and_center(contour)
                
                # 計算置信度
                area_ratio = area / (w * h)
                confidence = min(0.9, 0.4 + circularity * 0.3 + min(area_ratio * 5, 0.2))
                
                # 驗證檢測結果
                if _validate_detection({
                    'detected': True,
                    'bbox': bbox,
                    'center': center
                }, w, h):
                    valid_seals.append({
                        'bbox': bbox,
                        'center': center,
                        'confidence': confidence,
                        'circularity': circularity
                    })
        
        if not valid_seals:
            return []
        
        # 按置信度排序
        valid_seals.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 使用 NMS 過濾重疊的檢測結果（提高IoU閾值以保留更多重疊結果）
        filtered_seals = _apply_nms(valid_seals, iou_threshold=0.6)
        
        # 返回前 max_seals 個
        return filtered_seals[:max_seals]
        
    except Exception as e:
        return []


def _calculate_iou(bbox1: Dict, bbox2: Dict) -> float:
    """
    計算兩個邊界框的 IoU (Intersection over Union)
    
    Args:
        bbox1: 第一個邊界框 {'x': int, 'y': int, 'width': int, 'height': int}
        bbox2: 第二個邊界框
        
    Returns:
        IoU 值 (0-1)
    """
    x1_min, y1_min = bbox1['x'], bbox1['y']
    x1_max, y1_max = bbox1['x'] + bbox1['width'], bbox1['y'] + bbox1['height']
    
    x2_min, y2_min = bbox2['x'], bbox2['y']
    x2_max, y2_max = bbox2['x'] + bbox2['width'], bbox2['y'] + bbox2['height']
    
    # 計算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 計算並集
    area1 = bbox1['width'] * bbox1['height']
    area2 = bbox2['width'] * bbox2['height']
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def _apply_nms(seals: list, iou_threshold: float = 0.3) -> list:
    """
    應用非極大值抑制 (NMS) 過濾重疊的檢測結果
    
    Args:
        seals: 檢測結果列表，每個元素包含 'bbox', 'center', 'confidence'
        iou_threshold: IoU 閾值，超過此值視為重疊
        
    Returns:
        過濾後的檢測結果列表
    """
    if not seals:
        return []
    
    # 按置信度排序（降序）
    sorted_seals = sorted(seals, key=lambda x: x['confidence'], reverse=True)
    
    filtered_seals = []
    while sorted_seals:
        # 選擇置信度最高的
        best = sorted_seals.pop(0)
        filtered_seals.append(best)
        
        # 移除與當前最佳結果重疊的其他結果
        sorted_seals = [
            seal for seal in sorted_seals
            if _calculate_iou(best['bbox'], seal['bbox']) < iou_threshold
        ]
    
    return filtered_seals


def _crop_seal_with_margin(image: np.ndarray, bbox: Dict, circularity: float = 0.0) -> Tuple[np.ndarray, Dict]:
    """
    根據 bbox 裁切印鑑區域，添加足夠的邊距以支持旋轉
    
    Args:
        image: 原始圖像
        bbox: 邊界框 {'x': int, 'y': int, 'width': int, 'height': int}
        circularity: 圓度（0-1），用於判斷是否需要較大邊距
        
    Returns:
        (裁切後的圖像, 更新後的邊界框)
    """
    h, w = image.shape[:2]
    x = bbox['x']
    y = bbox['y']
    width = bbox['width']
    height = bbox['height']
    
    # 計算邊距
    if circularity > 0.7:
        # 圓形圖章：較小邊距
        radius = max(width, height) / 2.0
        margin = int(radius * 0.2)
    else:
        # 非圓形圖章：較大邊距（確保45度旋轉不會裁切到內容）
        # 對於45度旋轉，需要邊距 = max(width, height) * (sqrt(2) - 1) / 2
        margin = int(max(width, height) * 0.414)  # sqrt(2) - 1 ≈ 0.414
    
    # 確保邊距至少為10像素
    margin = max(10, margin)
    
    # 計算裁切區域（添加邊距）
    crop_x = max(0, x - margin)
    crop_y = max(0, y - margin)
    crop_width = min(w - crop_x, width + 2 * margin)
    crop_height = min(h - crop_y, height + 2 * margin)
    
    # 確保裁切區域有效
    if crop_width <= 0 or crop_height <= 0:
        # 如果計算失敗，使用原始 bbox
        crop_x = max(0, x)
        crop_y = max(0, y)
        crop_width = min(w - crop_x, width)
        crop_height = min(h - crop_y, height)
    
    # 裁切圖像
    cropped = image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
    
    # 更新邊界框（相對於裁切後的圖像）
    updated_bbox = {
        'x': x - crop_x,
        'y': y - crop_y,
        'width': width,
        'height': height
    }
    
    return cropped, updated_bbox


def _rotate_and_match(
    template_image: np.ndarray,
    seal_image: np.ndarray,
    rotation_range: float = 45.0,
    angle_step_coarse: float = 5.0,
    angle_step_fine: float = 2.0,
    angle_step_ultra_fine: float = 1.0
) -> Tuple[float, float]:
    """
    對單個裁切的印鑑區域進行旋轉搜索，找到與模板圖像最相似的角度
    
    Args:
        template_image: 參考圖像（已去背景）
        seal_image: 待匹配的印鑑圖像（已去背景）
        rotation_range: 旋轉角度範圍（度）
        angle_step_coarse: 粗搜索步長（度）
        angle_step_fine: 細搜索步長（度）
        angle_step_ultra_fine: 精細搜索步長（度）
        
    Returns:
        (最佳旋轉角度, 最佳相似度)
    """
    import sys
    from pathlib import Path
    core_path = Path(__file__).parent.parent.parent / "core"
    sys.path.insert(0, str(core_path))
    from seal_compare import SealComparator
    
    comparator = SealComparator()
    
    # 轉換為灰度圖
    if len(template_image.shape) == 3:
        template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template_image.copy()
    
    if len(seal_image.shape) == 3:
        seal_gray = cv2.cvtColor(seal_image, cv2.COLOR_BGR2GRAY)
    else:
        seal_gray = seal_image.copy()
    
    h, w = seal_gray.shape[:2]
    
    # 階段1：粗搜索（使用縮小圖像）
    scale_factor = 0.3
    small_h = int(h * scale_factor)
    small_w = int(w * scale_factor)
    small_template = cv2.resize(template_gray, 
                                (min(small_w, template_gray.shape[1]), 
                                 min(small_h, template_gray.shape[0])), 
                                interpolation=cv2.INTER_AREA)
    small_seal = cv2.resize(seal_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
    
    candidates = []
    for angle in np.arange(-rotation_range, rotation_range + angle_step_coarse, angle_step_coarse):
        if angle < -80 or angle > 80:
            continue
        
        # 應用旋轉
        center = (small_w // 2, small_h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(small_seal, M, (small_w, small_h),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255)
        
        # 計算相似度
        similarity = comparator._fast_rotation_match(small_template, rotated)
        candidates.append((angle, similarity))
    
    if not candidates:
        return 0.0, 0.0
    
    # 選擇最佳候選
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_angle_coarse = candidates[0][0]
    best_similarity_coarse = candidates[0][1]
    
    # 早期終止：如果相似度很高，跳過後續搜索
    if best_similarity_coarse > 0.9:
        return best_angle_coarse, best_similarity_coarse
    
    # 階段2：細搜索（在最佳角度附近）
    fine_range = 10.0
    best_angle = best_angle_coarse
    best_similarity = best_similarity_coarse
    
    for angle_offset in np.arange(-fine_range, fine_range + angle_step_fine, angle_step_fine):
        angle = best_angle_coarse + angle_offset
        if angle < -80 or angle > 80:
            continue
        
        # 應用旋轉到原始尺寸圖像
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(seal_gray, M, (w, h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=255)
        
        # 計算相似度
        similarity = comparator._fast_rotation_match(template_gray, rotated)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_angle = angle
    
    # 早期終止：如果相似度很高，跳過精細搜索
    if best_similarity > 0.95:
        return best_angle, best_similarity
    
    # 階段3：精細搜索（在最佳角度附近）
    ultra_fine_range = 2.0
    for angle_offset in np.arange(-ultra_fine_range, ultra_fine_range + angle_step_ultra_fine, angle_step_ultra_fine):
        angle = best_angle + angle_offset
        if angle < -80 or angle > 80:
            continue
        
        # 應用旋轉到原始尺寸圖像
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(seal_gray, M, (w, h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=255)
        
        # 計算相似度
        similarity = comparator._fast_rotation_match(template_gray, rotated)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_angle = angle
    
    return best_angle, best_similarity


def detect_seals_with_rotation_matching(
    image1_path: str,
    image2_path: str,
    rotation_range: float = 45.0,
    angle_step: float = 1.0,
    max_seals: int = 10,
    timeout: float = 30.0
) -> Dict:
    """
    檢測圖像2中與圖像1最相似的印鑑（考慮旋轉）
    
    Args:
        image1_path: 參考圖像路徑（模板）
        image2_path: 包含多個印鑑的圖像路徑
        rotation_range: 旋轉角度範圍（度），默認45度
        angle_step: 旋轉角度步長（度），默認1度（用於精細搜索）
        max_seals: 最大返回數量，默認10個
        timeout: 超時時間（秒），默認30秒
        
    Returns:
        包含檢測結果的字典：
        {
            'detected': bool,
            'matches': [
                {
                    'bbox': {'x': int, 'y': int, 'width': int, 'height': int},
                    'center': {'center_x': int, 'center_y': int, 'radius': float},
                    'rotation_angle': float,
                    'similarity': float,
                    'confidence': float
                },
                ...
            ],
            'count': int,
            'reason': str (如果失敗)
        }
    """
    start_time = time.time()
    
    try:
        # 驗證圖像文件
        if not Path(image1_path).exists():
            return {
                'detected': False,
                'matches': [],
                'count': 0,
                'reason': '圖像1文件不存在'
            }
        
        if not Path(image2_path).exists():
            return {
                'detected': False,
                'matches': [],
                'count': 0,
                'reason': '圖像2文件不存在'
            }
        
        # 載入圖像
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        
        if image1 is None:
            return {
                'detected': False,
                'matches': [],
                'count': 0,
                'reason': '無法讀取圖像1文件'
            }
        
        if image2 is None:
            return {
                'detected': False,
                'matches': [],
                'count': 0,
                'reason': '無法讀取圖像2文件'
            }
        
        # 檢查超時
        if time.time() - start_time > timeout:
            return {
                'detected': False,
                'matches': [],
                'count': 0,
                'reason': '檢測超時'
            }
        
        # 預處理圖像1（去背景）
        import sys
        from pathlib import Path
        core_path = Path(__file__).parent.parent.parent / "core"
        sys.path.insert(0, str(core_path))
        from seal_compare import SealComparator
        
        comparator = SealComparator()
        image1_processed = comparator._auto_detect_bounds_and_remove_background(image1)
        
        # 在圖像2中檢測所有可能的印鑑區域
        detection_result = detect_multiple_seals(image2_path, timeout=timeout - (time.time() - start_time), max_seals=50)
        
        if not detection_result.get('detected') or not detection_result.get('seals'):
            return {
                'detected': False,
                'matches': [],
                'count': 0,
                'reason': '未檢測到印鑑區域'
            }
        
        # 對每個檢測到的區域進行旋轉匹配
        matches = []
        h2, w2 = image2.shape[:2]
        
        for seal in detection_result['seals']:
            # 檢查超時
            if time.time() - start_time > timeout:
                break
            
            bbox = seal['bbox']
            center = seal['center']
            detection_confidence = seal.get('confidence', 0.5)
            
            # 計算圓度（從檢測結果中獲取，如果沒有則計算）
            circularity = seal.get('circularity', 0.0)
            if circularity == 0.0:
                # 簡單計算圓度（基於 bbox 的寬高比）
                aspect_ratio = min(bbox['width'], bbox['height']) / max(bbox['width'], bbox['height'])
                circularity = aspect_ratio  # 簡化處理
            
            # 裁切印鑑區域（添加邊距）
            try:
                cropped_seal, updated_bbox = _crop_seal_with_margin(image2, bbox, circularity)
                
                # 去背景處理
                cropped_seal_processed = comparator._auto_detect_bounds_and_remove_background(cropped_seal)
                
                # 旋轉匹配
                best_angle, best_similarity = _rotate_and_match(
                    image1_processed,
                    cropped_seal_processed,
                    rotation_range=rotation_range,
                    angle_step_coarse=5.0,
                    angle_step_fine=2.0,
                    angle_step_ultra_fine=angle_step
                )
                
                # 計算最終置信度（結合檢測置信度和相似度）
                final_confidence = (detection_confidence * 0.3 + best_similarity * 0.7)
                
                # 更新 bbox（考慮裁切偏移）
                final_bbox = {
                    'x': bbox['x'],
                    'y': bbox['y'],
                    'width': bbox['width'],
                    'height': bbox['height']
                }
                
                matches.append({
                    'bbox': final_bbox,
                    'center': center,
                    'rotation_angle': float(best_angle),
                    'similarity': float(best_similarity),
                    'confidence': float(final_confidence)
                })
                
            except Exception as e:
                # 跳過處理失敗的區域
                continue
        
        # 按相似度排序，選擇前 max_seals 個
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        top_matches = matches[:max_seals]
        
        return {
            'detected': len(top_matches) > 0,
            'matches': top_matches,
            'count': len(top_matches),
            'reason': None if len(top_matches) > 0 else '未找到匹配的印鑑'
        }
        
    except Exception as e:
        return {
            'detected': False,
            'matches': [],
            'count': 0,
            'reason': f'檢測過程出錯: {str(e)}'
        }
