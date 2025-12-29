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
import logging
import os
import json


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


def _is_debug_enabled() -> bool:
    v = os.getenv("SEAL_DETECTOR_DEBUG", "")
    return v.lower() in ("1", "true", "yes", "on")


def _get_debug_root_dir() -> Path:
    return Path(os.getenv("SEAL_DETECTOR_DEBUG_DIR", "/app/logs/detect_multiple_seals"))


def _safe_mkdir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Debug 輸出失敗不應影響主要功能
        pass


def _debug_write_image(debug_dir: Optional[Path], filename: str, image: np.ndarray) -> None:
    if debug_dir is None:
        return
    try:
        _safe_mkdir(debug_dir)
        cv2.imwrite(str(debug_dir / filename), image)
    except Exception:
        pass


def _debug_write_json(debug_dir: Optional[Path], filename: str, data: Dict) -> None:
    if debug_dir is None:
        return
    try:
        _safe_mkdir(debug_dir)
        (debug_dir / filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _draw_bboxes(image: np.ndarray, seals: list, color: Tuple[int, int, int]) -> np.ndarray:
    canvas = image.copy()
    for s in seals:
        try:
            b = s.get("bbox") or {}
            x, y, w, h = int(b.get("x", 0)), int(b.get("y", 0)), int(b.get("width", 0)), int(b.get("height", 0))
            if w <= 0 or h <= 0:
                continue
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        except Exception:
            continue
    return canvas


def _get_float_env(name: str, default: float) -> float:
    try:
        v = os.getenv(name)
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _get_int_env(name: str, default: int) -> int:
    try:
        v = os.getenv(name)
        if v is None or v == "":
            return default
        return int(v)
    except Exception:
        return default


def _red_mask_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """取得紅色印鑑候選 mask（255=前景）。"""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # 兩段紅色（OpenCV H: 0-179）
    lower1 = np.array([0, 60, 50], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([160, 60, 50], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    return cv2.bitwise_or(m1, m2)


def _connected_components_to_seals(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
    max_seals: int,
    *,
    base_min_area_ratio: float,
    min_fill_ratio: float,
    min_side: int = 10,
    size_filter: bool = True,
    debug_dir: Optional[Path] = None,
    debug_prefix: str = "components"
) -> list:
    """
    將二值 mask 轉成 seal bbox 候選（精度優先過濾）。
    mask: 255=前景
    """
    h, w = image_shape
    if mask is None or mask.size == 0:
        return []

    # 動態 min_area：max_seals 越大，min_area 越小（避免密集小印鑑被濾掉）
    scale = 10.0 / max(10, int(max_seals))
    min_area = int((w * h) * base_min_area_ratio * scale)
    min_area = max(120, min_area)
    max_area = int((w * h) * 0.15)

    num, labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    seals = []

    for label in range(1, num):
        x, y, bw, bh, area = stats[label]
        if area < min_area or area > max_area:
            continue
        if bw < min_side or bh < min_side:
            continue

        aspect = bw / max(1, bh)
        if aspect < 0.25 or aspect > 4.0:
            continue

        bbox_area = bw * bh
        fill_ratio = area / max(1, bbox_area)
        if fill_ratio < min_fill_ratio:
            continue

        bbox = {"x": int(x), "y": int(y), "width": int(bw), "height": int(bh)}
        center = {"center_x": int(x + bw / 2), "center_y": int(y + bh / 2), "radius": float(min(bw, bh) / 2)}
        # confidence：以填充率為主（精度優先）
        confidence = min(0.95, 0.35 + 0.6 * min(1.0, fill_ratio))

        # 復用既有的 bbox/center 合法性檢查（邊界/尺寸）
        if _validate_detection({"detected": True, "bbox": bbox, "center": center}, w, h):
            seals.append({
                "bbox": bbox,
                "center": center,
                "confidence": float(confidence),
                "_area": int(area),
                "_bbox_area": int(bbox_area),
                "_fill_ratio": float(fill_ratio),
            })

    # 若候選太多，依「尺寸一致性」再過濾一輪（精度優先：去掉大量小碎片/大塊雜訊）
    if size_filter and len(seals) > max(80, int(max_seals * 1.2)):
        top = sorted(seals, key=lambda s: s.get("confidence", 0.0), reverse=True)[:min(250, len(seals))]
        bbox_areas = [s.get("_bbox_area", 0) for s in top if s.get("_bbox_area", 0) > 0]
        if bbox_areas:
            median_bbox_area = float(np.median(np.array(bbox_areas, dtype=np.float32)))
            lo = median_bbox_area * _get_float_env("SEAL_MULTI_SIZE_MEDIAN_LO", 0.45)
            hi = median_bbox_area * _get_float_env("SEAL_MULTI_SIZE_MEDIAN_HI", 2.8)
            seals = [s for s in seals if lo <= float(s.get("_bbox_area", 0)) <= hi]
            _debug_write_json(debug_dir, f"{debug_prefix}_size_filter.json", {
                "median_bbox_area": median_bbox_area,
                "lo": lo,
                "hi": hi,
                "kept_after_size_filter": len(seals)
            })

    # 清理 debug 欄位
    for s in seals:
        s.pop("_area", None)
        s.pop("_bbox_area", None)
        s.pop("_fill_ratio", None)

    # 置信度排序 + NMS（CC 結果通常不重疊，但保留以防後續拆分/合併）
    seals.sort(key=lambda s: s.get("confidence", 0.0), reverse=True)
    try:
        seals_nms = _apply_nms(seals, iou_threshold=0.35)
    except Exception:
        seals_nms = seals

    final = seals_nms[:max_seals]

    _debug_write_json(debug_dir, f"{debug_prefix}_stats.json", {
        "cc_total_labels": int(num - 1),
        "min_area": int(min_area),
        "max_area": int(max_area),
        "kept_before_nms": int(len(seals)),
        "kept_after_nms": int(len(seals_nms)),
        "returned": int(len(final))
    })
    if debug_dir is not None:
        try:
            _debug_write_image(debug_dir, f"{debug_prefix}_bboxes.jpg", _draw_bboxes(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), final, (0, 255, 0)))
        except Exception:
            pass

    return final


def _mask_contours_to_seals(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
    max_seals: int,
    *,
    base_min_area_ratio: float,
    min_fill_ratio: float,
    min_side: int = 10,
    debug_dir: Optional[Path] = None,
    debug_prefix: str = "mask_contours"
) -> list:
    """用二值 mask 的外輪廓來取 seal bbox（可把同一印鑑內的多筆畫/碎片合併成一個 bbox）。"""
    h, w = image_shape
    if mask is None or mask.size == 0:
        return []

    scale = 10.0 / max(10, int(max_seals))
    min_area = int((w * h) * base_min_area_ratio * scale)
    min_area = max(200, min_area)
    max_area = int((w * h) * 0.25)

    m = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        _debug_write_json(debug_dir, f"{debug_prefix}_stats.json", {"contours_total": 0, "returned": 0})
        return []

    def split_watershed(mask_roi: np.ndarray) -> list:
        """
        用 distance transform + watershed 嘗試把黏在一起的大塊前景拆成多塊。
        回傳每一塊的 bbox（ROI 座標系）。
        """
        try:
            m = (mask_roi > 0).astype(np.uint8)
            if m.sum() == 0:
                return []

            dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
            if dist.max() <= 0:
                return []

            # sure foreground: 取距離峰值的高區域當作種子
            fg_thresh = float(_get_float_env("SEAL_MULTI_WS_FG_RATIO", 0.45)) * float(dist.max())
            _, sure_fg = cv2.threshold(dist, fg_thresh, 255, 0)
            sure_fg = sure_fg.astype(np.uint8)

            # markers
            n_markers, markers = cv2.connectedComponents(sure_fg)
            if n_markers <= 2:  # 太少種子，拆不開
                return []
            markers = markers + 1
            markers[m == 0] = 0

            # 用 dist 當作影像梯度（避免依賴原圖顏色）
            dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            ws_img = cv2.cvtColor(dist_norm, cv2.COLOR_GRAY2BGR)
            cv2.watershed(ws_img, markers)

            bboxes = []
            for lab in range(2, markers.max() + 1):
                comp = (markers == lab).astype(np.uint8)
                if comp.sum() == 0:
                    continue
                ys, xs = np.where(comp > 0)
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                bw, bh = (x1 - x0 + 1), (y1 - y0 + 1)
                if bw >= 5 and bh >= 5:
                    bboxes.append((x0, y0, bw, bh))
            return bboxes
        except Exception:
            return []

    candidates = []
    for c in contours:
        try:
            area = float(cv2.contourArea(c))
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < min_side or bh < min_side:
                continue
            bbox_area = float(bw * bh)
            fill_ratio = float(area / max(1.0, bbox_area))
            if fill_ratio < min_fill_ratio:
                continue
            candidates.append((c, area, x, y, bw, bh, fill_ratio))
        except Exception:
            continue

    if not candidates:
        _debug_write_json(debug_dir, f"{debug_prefix}_stats.json", {"contours_total": int(len(contours)), "kept": 0, "returned": 0})
        return []

    # 針對「過大輪廓」做 watershed 拆分（只在必要時啟用）
    areas = np.array([c[1] for c in candidates], dtype=np.float32)
    median_area = float(np.median(areas)) if areas.size else 0.0
    split_factor = _get_float_env("SEAL_MULTI_WS_SPLIT_FACTOR", 3.5)

    seals = []
    split_applied = 0
    for c, area, x, y, bw, bh, fill_ratio in candidates:
        do_split = (median_area > 0 and area > median_area * split_factor)
        if do_split:
            roi = m[y:y+bh, x:x+bw]
            split_boxes = split_watershed(roi)
            if len(split_boxes) >= 2:
                split_applied += 1
                for sx, sy, sbw, sbh in split_boxes:
                    bbox = {"x": int(x + sx), "y": int(y + sy), "width": int(sbw), "height": int(sbh)}
                    center = {"center_x": int(bbox["x"] + sbw / 2), "center_y": int(bbox["y"] + sbh / 2), "radius": float(min(sbw, sbh) / 2)}
                    confidence = min(0.92, 0.30 + 0.55)  # split 產物保守一點
                    if _validate_detection({"detected": True, "bbox": bbox, "center": center}, w, h):
                        seals.append({"bbox": bbox, "center": center, "confidence": float(confidence)})
                continue

        bbox = {"x": int(x), "y": int(y), "width": int(bw), "height": int(bh)}
        center = {"center_x": int(x + bw / 2), "center_y": int(y + bh / 2), "radius": float(min(bw, bh) / 2)}
        confidence = min(0.95, 0.35 + 0.6 * min(1.0, float(fill_ratio)))
        if _validate_detection({"detected": True, "bbox": bbox, "center": center}, w, h):
            seals.append({"bbox": bbox, "center": center, "confidence": float(confidence)})

    seals.sort(key=lambda s: s.get("confidence", 0.0), reverse=True)
    final = seals[:max_seals]

    _debug_write_json(debug_dir, f"{debug_prefix}_stats.json", {
        "contours_total": int(len(contours)),
        "candidates_kept": int(len(candidates)),
        "median_area": float(median_area),
        "watershed_split_applied": int(split_applied),
        "min_area": int(min_area),
        "max_area": int(max_area),
        "kept": int(len(seals)),
        "returned": int(len(final))
    })
    if debug_dir is not None:
        try:
            _debug_write_image(debug_dir, f"{debug_prefix}_bboxes.jpg", _draw_bboxes(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), final, (0, 255, 0)))
        except Exception:
            pass

    return final


def _detect_multiple_auto(image: np.ndarray, max_seals: int, debug_dir: Optional[Path]) -> Tuple[list, str]:
    """
    auto：顏色(紅) → 二值化連通元件 → 輪廓法 fallback
    回傳 (seals, method_name)
    """
    h, w = image.shape[:2]
    # base_min_area_ratio：越大越嚴格（去雜訊、但可能漏小印鑑）
    base_min_area_ratio_red = _get_float_env("SEAL_MULTI_BASE_MIN_AREA_RATIO_RED", 0.003)
    base_min_area_ratio_bin = _get_float_env("SEAL_MULTI_BASE_MIN_AREA_RATIO_BIN", 0.001)

    # fill_ratio：越大越嚴格（精度優先）
    min_fill_ratio_red = _get_float_env("SEAL_MULTI_MIN_FILL_RATIO_RED", 0.35)
    min_fill_ratio_bin = _get_float_env("SEAL_MULTI_MIN_FILL_RATIO_BIN", 0.28)

    min_side_red = _get_int_env("SEAL_MULTI_MIN_SIDE_RED", 14)
    min_side_bin = _get_int_env("SEAL_MULTI_MIN_SIDE_BIN", 14)

    # 1) 紅色分割（精度優先）
    red_mask = _red_mask_hsv(image)
    red_ratio = float(np.count_nonzero(red_mask)) / float(red_mask.size)
    _debug_write_json(debug_dir, "auto_meta.json", {
        "red_ratio": red_ratio,
        "base_min_area_ratio_red": base_min_area_ratio_red,
        "base_min_area_ratio_bin": base_min_area_ratio_bin,
        "max_seals": int(max_seals)
    })
    if debug_dir is not None:
        _debug_write_image(debug_dir, "red_mask_raw.png", red_mask)

    if red_ratio > _get_float_env("SEAL_MULTI_RED_RATIO_MIN", 0.001):
        k = _get_int_env("SEAL_MULTI_MORPH_KERNEL", 3)
        open_iters = _get_int_env("SEAL_MULTI_OPEN_ITERS_RED", 1)
        # 紅色 mask 用較強的 close，把同一印鑑內的碎片合併（避免一個印鑑被拆成多個 component）
        close_iters = _get_int_env("SEAL_MULTI_CLOSE_ITERS_RED", 2)
        kernel = np.ones((k, k), np.uint8)
        rm = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=open_iters)
        rm = cv2.morphologyEx(rm, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
        if debug_dir is not None:
            _debug_write_image(debug_dir, "red_mask.png", rm)
        red_seals = _mask_contours_to_seals(
            rm, (h, w), max_seals,
            base_min_area_ratio=base_min_area_ratio_red,
            min_fill_ratio=min_fill_ratio_red,
            min_side=min_side_red,
            debug_dir=debug_dir,
            debug_prefix="red_mask_contours"
        )
        # 若紅色策略有抓到，優先使用（精度優先）
        if red_seals:
            return red_seals, "auto_red"

    # 2) 二值化 + 連通元件（不依賴顏色）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu（假設前景較深，必要時自動反相）
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 前景太少/太多時嘗試反相（避免白底黑字或黑底白字問題）
    fg_ratio = float(np.count_nonzero(th == 0)) / float(th.size)
    if fg_ratio < 0.02 or fg_ratio > 0.85:
        th = cv2.bitwise_not(th)
    # 將前景統一成 255
    bin_mask = cv2.bitwise_not(th)
    k2 = _get_int_env("SEAL_MULTI_MORPH_KERNEL_BIN", 3)
    open_iters2 = _get_int_env("SEAL_MULTI_OPEN_ITERS_BIN", 1)
    close_iters2 = _get_int_env("SEAL_MULTI_CLOSE_ITERS_BIN", 1)
    kernel2 = np.ones((k2, k2), np.uint8)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel2, iterations=open_iters2)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel2, iterations=close_iters2)
    if debug_dir is not None:
        _debug_write_image(debug_dir, "binary_mask.png", bin_mask)

    bin_seals = _connected_components_to_seals(
        bin_mask, (h, w), max_seals,
        base_min_area_ratio=base_min_area_ratio_bin,
        min_fill_ratio=min_fill_ratio_bin,
        min_side=min_side_bin,
        debug_dir=debug_dir,
        debug_prefix="bin"
    )
    if bin_seals:
        return bin_seals, "auto_binary"

    # 3) fallback：原本輪廓法
    return [], "auto_none"


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
    debug_dir: Optional[Path] = None
    if _is_debug_enabled():
        try:
            base = Path(image_path).stem
            debug_dir = _get_debug_root_dir() / f"{base}_{int(start_time)}"
            _safe_mkdir(debug_dir)
            _debug_write_json(debug_dir, "meta.json", {"image_path": image_path, "timeout": timeout, "max_seals": max_seals})
        except Exception:
            debug_dir = None
    
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
        
        # Debug: 保存原圖（縮小以免檔案過大）
        if debug_dir is not None:
            try:
                h0, w0 = image.shape[:2]
                scale = min(1.0, 1600.0 / max(h0, w0))
                if scale < 1.0:
                    preview = cv2.resize(image, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
                else:
                    preview = image
                _debug_write_image(debug_dir, "input_preview.jpg", preview)
            except Exception:
                pass

        # 使用多印鑑檢測方法
        result = _detect_multiple_seals_fast(image, timeout - (time.time() - start_time), max_seals, debug_dir=debug_dir)
        
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


def _detect_multiple_seals_fast(
    image: np.ndarray,
    remaining_time: float,
    max_seals: int = 10,
    debug_dir: Optional[Path] = None
) -> Dict:
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
    
    method = os.getenv("SEAL_MULTI_METHOD", "auto").lower().strip()

    # auto/顏色/二值化 優先（精度優先），再 fallback 輪廓
    if remaining_time > 0.5 and method in ("auto", "red", "binary"):
        if method == "red":
            seals, used = _detect_multiple_auto(image, max_seals, debug_dir)
            seals = seals if used == "auto_red" else []
            used = "red"
        elif method == "binary":
            seals, used = _detect_multiple_auto(image, max_seals, debug_dir)
            seals = seals if used == "auto_binary" else []
            used = "binary"
        else:
            seals, used = _detect_multiple_auto(image, max_seals, debug_dir)

        if seals:
            _debug_write_json(debug_dir, "result.json", {"detected": True, "count": len(seals), "method": used})
            return {'detected': True, 'seals': seals, 'count': len(seals)}

    # 使用輪廓檢測方法檢測所有符合條件的印鑑（fallback / 或手動指定）
    if remaining_time > 1.0 and method in ("auto", "contours", "contour"):
        seals = _detect_multiple_by_contours(image, max_seals, debug_dir=debug_dir)
        if seals:
            _debug_write_json(debug_dir, "result.json", {"detected": True, "count": len(seals), "method": "contours"})
            return {'detected': True, 'seals': seals, 'count': len(seals)}
    
    # 如果沒有檢測到，返回空結果
    _debug_write_json(debug_dir, "result.json", {"detected": False, "count": 0, "method": "contours", "reason": "未檢測到印鑑"})
    return {
        'detected': False,
        'seals': [],
        'count': 0,
        'reason': '未檢測到印鑑'
    }


def _detect_multiple_by_contours(image: np.ndarray, max_seals: int = 10, debug_dir: Optional[Path] = None) -> list:
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

        # Debug: 輸出中間圖（灰階/邊緣）
        if debug_dir is not None:
            try:
                _debug_write_image(debug_dir, "gray.jpg", gray if len(gray.shape) == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY))
                _debug_write_image(debug_dir, "edges.png", edges)
            except Exception:
                pass
        
        # 查找輪廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            _debug_write_json(debug_dir, "stats.json", {"contours_total": 0, "valid_after_filters": 0, "nms_count": 0})
            return []
        
        # 篩選輪廓（基於面積和圓度）- 依 max_seals 動態調整最小面積（避免密集小印鑑被濾掉）
        valid_seals = []
        contours_total = len(contours)
        area_pass = 0
        circularity_pass = 0
        validate_pass = 0
        error_count = 0
        for contour in contours:
            try:
                area = cv2.contourArea(contour)
                base_min_area_ratio = _get_float_env("SEAL_MULTI_BASE_MIN_AREA_RATIO", 0.001)
                scale = 10.0 / max(10, int(max_seals))
                min_area = (w * h) * base_min_area_ratio * scale
                if area < min_area:
                    continue
                if area > (w * h * 0.95):  # 放寬最大面積限制（從90%提高到95%）
                    continue
                area_pass += 1
                circularity = _calculate_circularity(contour)
                if circularity > 0.05:  # 降低圓度要求（從0.2降到0.05）
                    circularity_pass += 1
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
                        validate_pass += 1
                        valid_seals.append({
                            'bbox': bbox,
                            'center': center,
                            'confidence': confidence,
                            'circularity': circularity
                        })
            except Exception as e:
                # 單個輪廓處理失敗時，記錄錯誤但繼續處理下一個輪廓
                error_count += 1
                # 如果錯誤太多（超過總輪廓數的50%），記錄警告但繼續處理
                if error_count > len(contours) * 0.5:
                    logging.warning(f"檢測過程中遇到較多錯誤 ({error_count}/{len(contours)})，但繼續處理剩餘輪廓")
                continue
        
        if not valid_seals:
            _debug_write_json(debug_dir, "stats.json", {
                "contours_total": contours_total,
                "area_pass": area_pass,
                "circularity_pass": circularity_pass,
                "validate_pass": validate_pass,
                "valid_after_filters": 0,
                "nms_count": 0
            })
            return []
        
        # 按置信度排序
        valid_seals.sort(key=lambda x: x['confidence'], reverse=True)

        # Debug: 畫出過濾後 bbox（NMS 前）
        if debug_dir is not None:
            try:
                _debug_write_image(debug_dir, "bboxes_pre_nms.jpg", _draw_bboxes(image, valid_seals[:min(len(valid_seals), 200)], (0, 255, 255)))
            except Exception:
                pass
        
        # 使用 NMS 過濾重疊的檢測結果（提高IoU閾值以保留更多重疊結果）
        try:
            filtered_seals = _apply_nms(valid_seals, iou_threshold=0.6)
        except Exception as e:
            # 如果 NMS 處理失敗，返回原始排序結果
            logging.warning(f"NMS 處理失敗: {str(e)}，返回原始檢測結果")
            filtered_seals = valid_seals
        
        # 返回前 max_seals 個
        final_seals = filtered_seals[:max_seals]

        _debug_write_json(debug_dir, "stats.json", {
            "contours_total": contours_total,
            "area_pass": area_pass,
            "circularity_pass": circularity_pass,
            "validate_pass": validate_pass,
            "valid_after_filters": len(valid_seals),
            "nms_count": len(filtered_seals),
            "returned": len(final_seals)
        })

        # Debug: 畫出最終 bbox（NMS 後）
        if debug_dir is not None:
            try:
                _debug_write_image(debug_dir, "bboxes_post_nms.jpg", _draw_bboxes(image, final_seals, (0, 255, 0)))
            except Exception:
                pass

        return final_seals
        
    except Exception as e:
        # 記錄錯誤但返回空列表（外層函數會處理）
        logging.error(f"多印鑑檢測過程出錯: {str(e)}")
        _debug_write_json(debug_dir, "error.json", {"error": str(e)})
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
    rotation_range: float = 15.0,
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
    rotation_range: float = 15.0,
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
        image1_processed, _ = comparator._auto_detect_bounds_and_remove_background(image1)
        
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
                cropped_seal_processed, _ = comparator._auto_detect_bounds_and_remove_background(cropped_seal)
                
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
