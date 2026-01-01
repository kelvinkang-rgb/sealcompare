import glob
from pathlib import Path

import cv2
import numpy as np

from core.seal_compare import SealComparator


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "background_removal" / "from_task_2704580d"


def _estimate_bg_ab(img_bgr: np.ndarray, edge_width: int) -> tuple[int, int]:
    h, w = img_bgr.shape[:2]
    ew = max(2, min(int(edge_width), h // 2, w // 2))
    edges = np.concatenate([
        img_bgr[:ew, :, :].reshape(-1, 3),
        img_bgr[-ew:, :, :].reshape(-1, 3),
        img_bgr[:, :ew, :].reshape(-1, 3),
        img_bgr[:, -ew:, :].reshape(-1, 3),
    ], axis=0)
    lab = cv2.cvtColor(edges.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    bg_a = int(np.median(lab[:, 1]))
    bg_b = int(np.median(lab[:, 2]))
    return bg_a, bg_b


def _proxy_ink_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    更寬鬆的 ink proxy（紅/藍自動選），用來定義「應保留的印泥像素」。
    這個 proxy 刻意比演算法寬鬆，以便抓到淡紅/淡藍。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_bgr.shape[:2]
    ew = max(3, min(h, w) // 30)
    bg_a, bg_b = _estimate_bg_ab(img_bgr, ew)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.int16)
    b = lab[:, :, 2].astype(np.int16)

    # 紅：delta_a >= 6（淡紅也能抓到）+ 寬鬆 HSV red
    delta_a = a - bg_a
    red_lab = ((delta_a >= 6) & (gray < 252)).astype(np.uint8) * 255
    r1 = cv2.inRange(hsv, np.array([0, 8, 15], np.uint8), np.array([15, 255, 255], np.uint8))
    r2 = cv2.inRange(hsv, np.array([165, 8, 15], np.uint8), np.array([179, 255, 255], np.uint8))
    red_hsv = cv2.bitwise_or(r1, r2)
    red = cv2.bitwise_or(red_lab, red_hsv)

    # 藍：bg_b - b >= 6（淡藍）+ 寬鬆 HSV blue
    delta_b = bg_b - b
    blue_lab = ((delta_b >= 6) & (gray < 252)).astype(np.uint8) * 255
    blue_hsv = cv2.inRange(hsv, np.array([90, 8, 15], np.uint8), np.array([140, 255, 255], np.uint8))
    blue = cv2.bitwise_or(blue_lab, blue_hsv)

    # 自動選：取像素數較多者
    if int(np.sum(blue > 0)) > int(np.sum(red > 0)):
        return blue
    return red


def test_background_removal_preserves_light_ink_pixels_from_task_2704580d():
    assert FIXTURES_DIR.exists(), f"fixtures not found: {FIXTURES_DIR}"
    files = sorted(glob.glob(str(FIXTURES_DIR / "*.png")))
    assert files, "no fixture png files found"

    comparator = SealComparator()

    for fp in files:
        img = cv2.imread(fp)
        assert img is not None, f"failed to read: {fp}"

        before = _proxy_ink_mask(img) > 0
        before_count = int(np.sum(before))
        assert before_count > 200, f"expected enough ink-like pixels in fixture: {Path(fp).name}"

        out, _timing = comparator._auto_detect_bounds_and_remove_background(img, return_timing=True)
        assert out is not None and out.size > 0

        after = _proxy_ink_mask(out) > 0
        after_count = int(np.sum(after))

        # 需求：寧可多留噪點，也不要吃掉淡紅/淡藍筆劃
        # 以 proxy ink mask 的像素數當量化指標：不可明顯下降
        assert after_count >= int(before_count * 0.98), (
            f"{Path(fp).name}: ink-like pixels dropped too much "
            f"before={before_count} after={after_count}"
        )


