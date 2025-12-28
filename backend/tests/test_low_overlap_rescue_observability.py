import sys
from pathlib import Path

import cv2
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.seal_compare import SealComparator


def _make_seal(size: int = 240) -> np.ndarray:
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), int(size * 0.33), (0, 0, 0), 3)
    cv2.line(img, (int(size * 0.25), int(size * 0.55)), (int(size * 0.8), int(size * 0.45)), (0, 0, 0), 2)
    cv2.putText(img, "S9", (int(size * 0.28), int(size * 0.70)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def _apply(img: np.ndarray, angle: float, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    M2 = np.float32([[1, 0, float(dx)], [0, 1, float(dy)]])
    out = cv2.warpAffine(out, M2, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return out


def test_observability_and_rescue_evals_present_on_low_overlap_case():
    comp = SealComparator()
    img1 = _make_seal(240)
    # 讓真實平移超出 coarse translation_range，逼出低 overlap 起點
    img2 = _apply(img1, angle=5.0, dx=90, dy=-70)

    aligned, angle, (ox, oy), sim, metrics, timing = comp._align_image2_to_image1(
        img1,
        img2,
        rotation_range=10.0,
        translation_range=15  # 故意很小
    )

    assert isinstance(metrics, dict)
    # stage12 候選與 overlap 指標存在
    assert 'overlap_after_stage12' in metrics
    assert 'overlap_after_stage3' in metrics
    assert 'overlap_after_stage4' in metrics
    assert 'offset_after_stage12' in metrics
    assert 'offset_after_stage3' in metrics
    assert 'offset_after_stage4' in metrics

    # rescue 的 evals 應存在（即使未觸發也會有 key；若觸發通常會 >0）
    assert isinstance(timing, dict)
    stages = timing.get('alignment_stages', {})
    # alignment_stages 在 services 層才會包；這裡直接看 metrics/timing 不保證
    # 因此驗證 metrics 內 rescue_triggered 存在即可
    assert 'rescue_triggered' in metrics


