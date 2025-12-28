import sys
from pathlib import Path

import cv2
import numpy as np

# pytest 在容器/本機的 rootdir 可能不同，明確把 backend 根目錄加到 sys.path
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.seal_compare import SealComparator


def _non_white_pixels(img: np.ndarray, thr: int = 245) -> int:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return int(np.sum(gray < thr))


def test_warp_affine_auto_canvas_keeps_content_vs_fixed_canvas():
    comp = SealComparator()
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    # 讓內容靠近邊緣，方便觀察裁切
    cv2.rectangle(img, (10, 80), (190, 120), (0, 0, 0), thickness=-1)

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 12.0, 1.0).astype(np.float32)
    # 故意做負平移，固定 canvas 一定會丟內容
    M[0, 2] += -70.0
    M[1, 2] += -40.0

    fixed = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    auto, canvas = comp._warp_affine_auto_canvas(img, M, include_sizes=[(w, h)], border_value=(255, 255, 255))

    assert _non_white_pixels(auto) > _non_white_pixels(fixed)
    assert 'shift_x' in canvas and 'shift_y' in canvas and 'w' in canvas and 'h' in canvas
    assert canvas['w'] >= w and canvas['h'] >= h


def test_align_image2_to_image1_exports_alignment_canvas_metrics():
    comp = SealComparator()
    img1 = np.full((256, 256, 3), 255, dtype=np.uint8)
    cv2.circle(img1, (128, 128), 60, (0, 0, 0), 3)
    cv2.putText(img1, "A7", (70, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    # forward transform 產生 img2（故意讓對齊需要負平移）
    angle = 8.0
    dx = 70
    dy = -50
    center = (128, 128)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img2 = cv2.warpAffine(img1, M, (256, 256), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    M2 = np.float32([[1, 0, float(dx)], [0, 1, float(dy)]])
    img2 = cv2.warpAffine(img2, M2, (256, 256), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    aligned, best_angle, (ox, oy), sim, metrics, timing = comp._align_image2_to_image1(
        img1,
        img2,
        rotation_range=15.0,
        translation_range=120
    )

    assert isinstance(metrics, dict)
    assert 'alignment_canvas' in metrics
    canvas = metrics['alignment_canvas']
    assert set(canvas.keys()) >= {'shift_x', 'shift_y', 'w', 'h'}
    assert aligned is not None and aligned.size > 0


