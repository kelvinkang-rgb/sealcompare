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
    img = np.full((size, size), 255, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), int(size * 0.35), 0, thickness=3)
    cv2.line(img, (int(size * 0.30), int(size * 0.40)), (int(size * 0.75), int(size * 0.55)), 0, 2)
    cv2.putText(img, "K3", (int(size * 0.28), int(size * 0.66)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)
    return img


def _apply_transform(img: np.ndarray, angle_deg: float, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M_rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M_rot, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    M_trans = np.float32([[1, 0, float(dx)], [0, 1, float(dy)]])
    transformed = cv2.warpAffine(rotated, M_trans, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return transformed


def test_translation_rescue_increases_overlap_ratio_on_big_miss():
    comp = SealComparator()
    img1 = _make_synthetic_seal(256)

    # 先用真實轉換生成 img2（forward transform）
    gt_angle = 5.0
    gt_dx = 20
    gt_dy = -15
    img2 = _apply_transform(img1, gt_angle, gt_dx, gt_dy)

    # 對齊時需要反向參數
    angle_align = -gt_angle
    dx_align = -gt_dx
    dy_align = -gt_dy

    # 人工製造 coarse 平移大錯峰（偏 80px）
    wrong_dx = dx_align + 80
    wrong_dy = dy_align - 60

    ov_before = comp._fast_overlap_ratio(img1, img2, angle_align, wrong_dx, wrong_dy, scale=0.3)
    rx, ry, ob, oa, timing = comp._translation_rescue_search(
        img1, img2,
        angle=angle_align,
        initial_offset_x=wrong_dx,
        initial_offset_y=wrong_dy,
        overlap_threshold=0.6,
        radius=120,
        steps=(8, 3, 1),
        scale=0.3
    )

    # overlap 應該顯著提升
    assert abs(ob - ov_before) < 1e-6
    assert oa > ob + 0.2

    # 救援後應更靠近真值（允許粗到細搜尋誤差）
    assert abs(rx - dx_align) < abs(wrong_dx - dx_align)
    assert abs(ry - dy_align) < abs(wrong_dy - dy_align)
    assert "stage3_translation_rescue_total" in timing


