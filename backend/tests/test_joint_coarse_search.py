import sys
from pathlib import Path

import cv2
import numpy as np

# pytest 在容器/本機的 rootdir 可能不同，明確把 backend 根目錄加到 sys.path
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.seal_compare import SealComparator


def _make_synthetic_seal(size: int = 256) -> np.ndarray:
    """建立可重現的合成印面（黑色內容 + 白底），避免外部檔案依賴。"""
    img = np.full((size, size), 255, dtype=np.uint8)

    # 外框圓
    cv2.circle(img, (size // 2, size // 2), int(size * 0.35), 0, thickness=3)
    # 內部幾條線，增加方向性避免旋轉對稱造成不確定
    cv2.line(img, (int(size * 0.35), int(size * 0.55)), (int(size * 0.70), int(size * 0.35)), 0, 2)
    cv2.line(img, (int(size * 0.30), int(size * 0.35)), (int(size * 0.55), int(size * 0.70)), 0, 2)

    # 加一些文字形狀（opencv內建字型）
    cv2.putText(img, "A9", (int(size * 0.32), int(size * 0.62)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)
    return img


def _apply_transform(img: np.ndarray, angle_deg: float, dx: int, dy: int) -> np.ndarray:
    """以既知旋轉+平移產生 img2（與對齊流程一致：先旋轉再平移）。"""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M_rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M_rot, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    M_trans = np.float32([[1, 0, float(dx)], [0, 1, float(dy)]])
    transformed = cv2.warpAffine(rotated, M_trans, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return transformed


def test_joint_coarse_search_multiscale_recovers_transform_reasonably():
    comp = SealComparator()

    img1 = _make_synthetic_seal(256)

    # ground truth：注意這是「把 img2 對齊到 img1」所需反向的參數
    # 我們先把 img1 生成 img2：先旋轉 +7.0 度，再平移 (dx, dy) = (18, -12)
    gt_angle_forward = 7.0
    gt_dx_forward = 18
    gt_dy_forward = -12
    img2 = _apply_transform(img1, gt_angle_forward, gt_dx_forward, gt_dy_forward)

    # 對齊時應回復到接近 -7 度，且平移接近 (-18, +12)
    expected_angle = -gt_angle_forward
    expected_dx = -gt_dx_forward
    expected_dy = -gt_dy_forward

    best_angle, best_dx, best_dy, best_sim, timing = comp._joint_coarse_search_multiscale(
        img1_gray=img1,
        img2_gray=img2,
        rotation_range=15.0,
        translation_range=256,
        pyramid_scales=(0.25, 0.5),
        top_k=5,
    )

    # coarse 搜尋允許較大誤差，後續階段3/4會精細化
    assert abs(best_angle - expected_angle) <= 6.0
    assert abs(best_dx - expected_dx) <= 25
    assert abs(best_dy - expected_dy) <= 25
    assert best_sim >= 0.2
    assert "stage12_joint_coarse_total" in timing


