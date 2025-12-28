import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.seal_compare import SealComparator


def _make_asymmetric_seal(size: int = 240) -> np.ndarray:
    """近似對稱但帶一個小破壞對稱特徵，讓 overlap 能分出 +θ/-θ。"""
    img = np.full((size, size), 255, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), int(size * 0.33), 0, 3)
    cv2.putText(img, "R", (int(size * 0.60), int(size * 0.55)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
    # 破壞對稱：右上角加一個小點
    cv2.circle(img, (int(size * 0.75), int(size * 0.25)), 3, 0, -1)
    return img


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def test_angle_sign_disambiguation_flips_when_sign_is_wrong():
    comp = SealComparator()
    img1 = _make_asymmetric_seal(240)

    # 生成 img2：先把 img1 旋轉 -12 度
    # 正確的對齊應該是 angle=+12（把 img2 轉回去）
    img2 = _rotate(img1, -12.0)

    wrong_angle = -12.0
    a2, dx2, dy2, m = comp._angle_sign_disambiguation(
        img1,
        img2,
        angle=wrong_angle,
        offset_x=0,
        offset_y=0,
        scale=0.3,
        improve_threshold=0.05,
        rescue_radius=40,
        rescue_steps=(6, 2, 1)
    )

    assert m.get('angle_sign_check_triggered') is True
    assert m.get('angle_sign_flipped') is True
    assert abs(a2 - 12.0) < 1e-6
    # 合成圖的重疊率提升幅度依內容而異，重點是「翻角度後 overlap 明顯變好」
    assert m.get('overlap_after_sign_check', 0.0) > m.get('overlap_before_sign_check', 0.0) + 0.05


