import cv2
import numpy as np

from core.seal_compare import SealComparator


def _warp_affine_translate(src: np.ndarray, offset_x: int, offset_y: int) -> np.ndarray:
    h, w = src.shape[:2]
    M = np.float32([[1, 0, float(offset_x)], [0, 1, float(offset_y)]])
    return cv2.warpAffine(src, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def test_shift_gray_with_border_matches_warp_affine_translation():
    comp = SealComparator()

    # 用可辨識的圖樣（非全白/非對稱），避免測試退化
    h, w = 64, 80
    src = np.full((h, w), 255, dtype=np.uint8)
    src[10:20, 15:25] = 0
    src[40:55, 5:18] = 30
    src[30, 60:78] = 120

    offsets = [
        (0, 0),
        (5, 0),
        (0, 7),
        (-6, 0),
        (0, -9),
        (12, -8),
        (-15, 11),
        (200, 0),    # 完全移出
        (0, -200),   # 完全移出
    ]

    buf = np.empty_like(src)
    for ox, oy in offsets:
        expected = _warp_affine_translate(src, ox, oy)
        actual = comp._shift_gray_with_border(src, ox, oy, out=buf, fill_value=255).copy()
        assert np.array_equal(actual, expected), f"mismatch at offset ({ox},{oy})"


