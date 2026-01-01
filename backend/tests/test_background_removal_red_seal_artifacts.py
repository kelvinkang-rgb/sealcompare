import glob
from pathlib import Path

import cv2
import numpy as np

from core.seal_compare import SealComparator


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "background_removal" / "from_task_c685fc4b"


def _border_pixels(gray: np.ndarray, edge_width: int) -> np.ndarray:
    h, w = gray.shape[:2]
    ew = max(2, min(edge_width, h // 2, w // 2))
    return np.concatenate([
        gray[:ew, :].ravel(),
        gray[-ew:, :].ravel(),
        gray[:, :ew].ravel(),
        gray[:, -ew:].ravel(),
    ])


def _red_mask_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    回傳 uint8 mask (0/255)；用「相對背景的色偏」當 proxy，避免淡紅因固定閾值而漏掉。
    這裡的 proxy 目標是：標記『看起來像紅印泥』的像素，供「保留率」檢查。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 背景估計：用邊緣像素的 Lab 中位數（抗噪）
    h, w = img_bgr.shape[:2]
    ew = max(3, min(h, w) // 30)
    edges = np.concatenate([
        img_bgr[:ew, :, :].reshape(-1, 3),
        img_bgr[-ew:, :, :].reshape(-1, 3),
        img_bgr[:, :ew, :].reshape(-1, 3),
        img_bgr[:, -ew:, :].reshape(-1, 3),
    ], axis=0)
    edges_lab = cv2.cvtColor(edges.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    bg_a = int(np.median(edges_lab[:, 1]))

    # HSV（寬鬆）+ Lab delta_a（核心）
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 10, 20], dtype=np.uint8)
    upper1 = np.array([15, 255, 255], dtype=np.uint8)
    lower2 = np.array([165, 10, 20], dtype=np.uint8)
    upper2 = np.array([179, 255, 255], dtype=np.uint8)
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    red_hsv = cv2.bitwise_or(m1, m2)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.int16)
    delta_a = a - bg_a
    red_lab = ((delta_a >= 8) & (gray < 252)).astype(np.uint8) * 255

    return cv2.bitwise_or(red_hsv, red_lab)


def test_background_removal_reduces_border_dark_artifacts_and_preserves_red_strokes():
    assert FIXTURES_DIR.exists(), f"fixtures not found: {FIXTURES_DIR}"

    files = sorted(glob.glob(str(FIXTURES_DIR / "*.png")))
    assert files, "no fixture png files found"

    comparator = SealComparator()

    for fp in files:
        img = cv2.imread(fp)
        assert img is not None, f"failed to read: {fp}"

        before_red_u8 = _red_mask_bgr(img)
        before_red = before_red_u8 > 0
        before_red_count = int(before_red.sum())
        assert before_red_count > 0, f"expected red pixels in fixture: {fp}"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 低飽和 + 偏暗：掃描黑線/摺痕陰影的 proxy（排除白底、也排除紅筆劃）
        k = np.ones((3, 3), np.uint8)
        before_red_dil = cv2.dilate(before_red_u8, k, iterations=2) > 0
        before_artifact = (hsv[:, :, 1] < 60) & (gray < 220) & (~before_red_dil)
        before_artifact_ratio = float(np.mean(before_artifact))

        out, _timing = comparator._auto_detect_bounds_and_remove_background(img, return_timing=True)
        assert out is not None and out.size > 0
        out_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        after_red_u8 = _red_mask_bgr(out)
        after_red = after_red_u8 > 0
        out_hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        after_red_dil = cv2.dilate(after_red_u8, k, iterations=2) > 0
        after_artifact = (out_hsv[:, :, 1] < 60) & (out_gray < 220) & (~after_red_dil)
        after_artifact_ratio = float(np.mean(after_artifact))

        # 黑線/陰影殘留應下降：
        # - 若原本殘留明顯（>=1%），至少下降 40%
        # - 若原本幾乎沒有殘留，至少不能顯著變差
        if before_artifact_ratio >= 0.01:
            assert after_artifact_ratio <= before_artifact_ratio * 0.6 + 1e-6, (
                f"{Path(fp).name}: artifact_ratio not improved enough "
                f"before={before_artifact_ratio:.3f} after={after_artifact_ratio:.3f}"
            )
        else:
            assert after_artifact_ratio <= before_artifact_ratio + 0.002, (
                f"{Path(fp).name}: artifact_ratio got worse "
                f"before={before_artifact_ratio:.4f} after={after_artifact_ratio:.4f}"
            )

        # 紅色筆劃應保留（以紅像素數 proxy，不可大量流失）
        after_red_count = int(after_red.sum())
        assert after_red_count >= int(before_red_count * 0.95), (
            f"{Path(fp).name}: red pixels dropped too much "
            f"before={before_red_count} after={after_red_count}"
        )


