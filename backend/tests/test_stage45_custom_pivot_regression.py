import sys
from pathlib import Path

import numpy as np
import cv2

# pytest 在容器/本機的 rootdir 可能不同，明確把 backend 根目錄加到 sys.path
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.seal_compare import SealComparator  # noqa: E402


def _make_offset_stamp_image(h: int, w: int, *, stamp_size: int = 70, offset: tuple[int, int] = (-60, 40)) -> np.ndarray:
    """
    建立一張白底灰階影像，放一個黑色方章在「偏離影像中心」的位置。
    這用來驗證：如果旋轉支點用影像中心，90 度旋轉後會引入很大的等效平移。
    """
    img = np.full((h, w), 255, dtype=np.uint8)
    cx = w // 2 + int(offset[0])
    cy = h // 2 + int(offset[1])

    half = stamp_size // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(w, cx + half)
    y1 = min(h, cy + half)
    img[y0:y1, x0:x1] = 0
    return img


def test_stage45_only_uses_bbox_center_pivot_and_converges_for_offset_stamp():
    """
    回歸測試（不 mock）：
    - 建立「偏置印鑑」的 image1
    - 令 image2 為 image1 的 90 度旋轉版本（模擬右角候選輸入）
    - stage45_only 不跑 stage1~3，且 stage5 稀疏平移只在 ±10px 內採樣
      若支點錯（影像中心），等效平移通常會 > 10px 而漏解
      改用 bbox 中心支點後，等效平移顯著變小，stage4/5 可收斂到高相似度
    """
    comp = SealComparator(threshold=0.5)

    img1 = _make_offset_stamp_image(256, 256, stamp_size=70, offset=(-60, 40))
    img2 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

    aligned, angle, (ox, oy), sim, metrics, timing = comp._align_image2_to_image1_stage45_only(
        img1,
        img2,
        rotation_range=15.0,
        translation_range=256,
    )

    assert isinstance(timing, dict)
    assert isinstance(metrics, dict)
    assert metrics.get("stage45_rotation_center_mode") == "bbox_center"
    rc = metrics.get("stage45_rotation_center") or {}
    assert "x" in rc and "y" in rc

    # 新行為：stage45_only 會用 pivot1 - pivot2 初始化 offset，先處理 30~100px 級別的大平移
    p1 = metrics.get("stage45_pivot1") or {}
    p2 = metrics.get("stage45_pivot2") or {}
    init = metrics.get("stage45_initial_offset_from_pivot") or {}
    assert "x" in p1 and "y" in p1
    assert "x" in p2 and "y" in p2
    assert "x" in init and "y" in init

    expected_x = int(round(float(p1["x"]) - float(p2["x"])))
    expected_y = int(round(float(p1["y"]) - float(p2["y"])))
    assert int(init["x"]) == expected_x
    assert int(init["y"]) == expected_y

    # 驗收：offset 需要接近 pivot 初始化（再由 stage4/5 做 1px/0.2° 微調）
    assert abs(int(ox) - expected_x) <= 5
    assert abs(int(oy) - expected_y) <= 5

    # 相似度應顯著提升（合成方塊對稱性高，門檻放寬以避免對稱造成分數波動）
    assert float(sim) > 0.90

