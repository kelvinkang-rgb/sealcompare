import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.seal_compare import SealComparator


def _make_asymmetric_seal(size: int = 260) -> np.ndarray:
    img = np.full((size, size), 255, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), int(size * 0.34), 0, 3)
    cv2.putText(img, "A", (int(size * 0.35), int(size * 0.62)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA)
    cv2.circle(img, (int(size * 0.78), int(size * 0.24)), 4, 0, -1)
    return img


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _translate(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.float32([[1, 0, float(dx)], [0, 1, float(dy)]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def test_angle_sign_disambiguation_flips_even_when_raw_not_better():
    """
    回歸測試：以前的實作只看「同一 dx/dy 下的 raw overlap」，
    可能因為翻號後需要小幅平移才會變好而錯過翻號。
    這裡刻意把初始 offset 設在「wrong sign 的局部最佳」，確保 raw flip 不會更好，
    但 flip + micro-rescue 後會更好，應該翻號。
    """
    comp = SealComparator()
    img1 = _make_asymmetric_seal(260)

    # img2 = rotate(-12) + translate(+18,-10) ; correct align should be +12 with roughly (-18,+10)
    img2 = _translate(_rotate(img1, -12.0), dx=18, dy=-10)

    wrong_angle = -12.0

    # 找到 wrong sign 下的局部最佳 offset（作為「陷阱」初始點）
    ox, oy, _, ov_wrong_best, _ = comp._translation_rescue_search(
        img1, img2, angle=wrong_angle, initial_offset_x=0, initial_offset_y=0,
        overlap_threshold=1.0, radius=40, steps=(6, 2, 1), scale=0.3
    )

    # 這個 offset 在 flip raw 時可能不會更好（舊版會因此不翻號）
    a2, dx2, dy2, m = comp._angle_sign_disambiguation(
        img1, img2, angle=wrong_angle, offset_x=int(ox), offset_y=int(oy),
        scale=0.3, improve_threshold=0.05, rescue_radius=40, rescue_steps=(6, 2, 1)
    )

    assert m.get('angle_sign_check_triggered') is True
    assert m.get('angle_sign_flipped') is True
    assert abs(float(a2) - 12.0) < 1e-6


