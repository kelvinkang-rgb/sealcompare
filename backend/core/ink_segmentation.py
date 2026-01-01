from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class InkSegmentationResult:
    color: Optional[str]  # "red" | "blue" | None
    mask_u8: np.ndarray  # uint8 mask (0/255)
    ratio: float


def _edge_pixels(img: np.ndarray, edge_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    ew = max(2, min(int(edge_width), h // 2, w // 2))
    if ew <= 0:
        return img.reshape(-1, img.shape[-1]) if img.ndim == 3 else img.reshape(-1, 1)
    if img.ndim == 2:
        img = img[:, :, None]
    return np.concatenate(
        [
            img[:ew, :, :].reshape(-1, 3),
            img[-ew:, :, :].reshape(-1, 3),
            img[:, :ew, :].reshape(-1, 3),
            img[:, -ew:, :].reshape(-1, 3),
        ],
        axis=0,
    )


def estimate_bg_lab(img_bgr: np.ndarray, edge_width: Optional[int] = None) -> Tuple[float, float, float]:
    """
    用邊緣像素估背景色（Lab），以 median 抗噪。
    回傳 (L, a, b) 浮點。
    """
    if img_bgr is None or img_bgr.size == 0:
        return (0.0, 128.0, 128.0)
    h, w = img_bgr.shape[:2]
    ew = edge_width if edge_width is not None else max(3, min(h, w) // 30)
    edges = _edge_pixels(img_bgr, ew).astype(np.uint8)
    lab = cv2.cvtColor(edges.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    med = np.median(lab, axis=0)
    return float(med[0]), float(med[1]), float(med[2])


def _cleanup_mask(mask_u8: np.ndarray, gray: np.ndarray) -> np.ndarray:
    if mask_u8 is None or mask_u8.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)
    k = 3 if min(gray.shape[:2]) < 220 else 5
    kernel = np.ones((k, k), np.uint8)
    out = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)

    # 讓筆劃更連續：把「靠近 ink」且不是高亮背景的像素納入（保護淡色筆劃）
    dil = cv2.dilate(out, kernel, iterations=1)
    near = (dil > 0) & (gray < 252)
    out = np.where(near, 255, out).astype(np.uint8)

    # 過濾散落噪點：保留主要連通元件（但優先保留細筆劃，門檻要夠寬鬆）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        max_area = int(areas.max()) if areas.size else 0
        keep = np.zeros_like(out, dtype=np.uint8)
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if max_area > 0 and area >= max(40, int(max_area * 0.005)):
                keep[labels == i] = 255
        out = keep
    return out


def red_ink_mask(img_bgr: np.ndarray, bg_lab: Tuple[float, float, float]) -> np.ndarray:
    """
    偵測紅色印泥前景（0/255 mask）。
    核心：用相對背景的 a-channel 偏移抓淡紅，同時保留 HSV 紅色範圍作為補強。
    """
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.int16)
    bga = int(round(bg_lab[1]))

    # HSV 紅色兩段（放寬 S 以涵蓋淡紅）
    lower1 = np.array([0, 10, 20], dtype=np.uint8)
    upper1 = np.array([15, 255, 255], dtype=np.uint8)
    lower2 = np.array([165, 10, 20], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    red_hsv = cv2.bitwise_or(m1, m2)

    # Lab：相對背景的紅度偏移（淡紅也會有 a 增量）
    delta_a = a - bga
    red_lab = ((delta_a >= 6) & (gray < 253)).astype(np.uint8) * 255

    raw = cv2.bitwise_or(red_hsv, red_lab)
    return _cleanup_mask(raw, gray)


def blue_ink_mask(img_bgr: np.ndarray, bg_lab: Tuple[float, float, float]) -> np.ndarray:
    """
    偵測藍色印泥前景（0/255 mask）。
    核心：用相對背景的 b-channel 偏移抓淡藍（藍偏向 b 下降），HSV 藍色範圍補強。
    """
    if img_bgr is None or img_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    b = lab[:, :, 2].astype(np.int16)
    bgb = int(round(bg_lab[2]))

    lower = np.array([90, 10, 20], dtype=np.uint8)
    upper = np.array([140, 255, 255], dtype=np.uint8)
    blue_hsv = cv2.inRange(hsv, lower, upper)

    # Lab：相對背景的藍度偏移（藍偏向 b 降低）
    delta_b = bgb - b
    blue_lab = ((delta_b >= 6) & (gray < 253)).astype(np.uint8) * 255

    raw = cv2.bitwise_or(blue_hsv, blue_lab)
    return _cleanup_mask(raw, gray)


def select_ink_color_and_mask(
    img_bgr: np.ndarray,
    *,
    min_ratio: float = 0.004,
) -> InkSegmentationResult:
    """
    自動在 red / blue 之間選擇前景 mask。
    若兩者都不足以信任（面積比例太低），回傳 color=None、mask=全0。
    """
    if img_bgr is None or img_bgr.size == 0:
        return InkSegmentationResult(color=None, mask_u8=np.zeros((1, 1), dtype=np.uint8), ratio=0.0)

    bg_lab = estimate_bg_lab(img_bgr)
    red = red_ink_mask(img_bgr, bg_lab)
    blue = blue_ink_mask(img_bgr, bg_lab)

    h, w = red.shape[:2]
    denom = float(h * w) if h > 0 and w > 0 else 1.0
    red_ratio = float(np.sum(red > 0)) / denom
    blue_ratio = float(np.sum(blue > 0)) / denom

    if red_ratio < min_ratio and blue_ratio < min_ratio:
        return InkSegmentationResult(color=None, mask_u8=np.zeros_like(red, dtype=np.uint8), ratio=max(red_ratio, blue_ratio))

    if red_ratio >= blue_ratio:
        return InkSegmentationResult(color="red", mask_u8=red, ratio=red_ratio)
    return InkSegmentationResult(color="blue", mask_u8=blue, ratio=blue_ratio)


