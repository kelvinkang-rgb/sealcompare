import cv2
import numpy as np

from core.seal_compare import SealComparator


def _make_stamp_bgr(size: int = 220) -> np.ndarray:
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.line(img, (40, 30), (40, size - 30), (0, 0, 0), 12)
    cv2.line(img, (40, size - 35), (size - 55, size - 35), (0, 0, 0), 12)
    cv2.circle(img, (size - 65, 70), 10, (0, 0, 0), -1)
    cv2.line(img, (85, 55), (135, 55), (0, 0, 0), 8)
    return img


def test_right_angle_candidates_are_skipped_when_structure_similarity_meets_threshold():
    # threshold 來自「本次請求」：這裡用 0.5（50%）
    comp = SealComparator(threshold=0.5)

    img1 = _make_stamp_bgr()
    img2 = img1.copy()

    # 造一個「深淺變化」版本：讓 _fast_rotation_match 可能下降，但結構應仍一致
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ink = gray < 250
    img2[ink] = np.clip(img2[ink].astype(np.int16) + 90, 0, 255).astype(np.uint8)

    aligned, angle, (ox, oy), sim, metrics, timing = comp._align_image2_to_image1(
        img1, img2, rotation_range=15.0, translation_range=100
    )

    assert isinstance(metrics, dict)
    assert metrics.get("right_angle_candidates_skipped_by_structure_similarity") is True
    assert metrics.get("right_angle_candidates_tried") == [0]
    assert metrics.get("right_angle_fallback_used") is False
    assert float(metrics.get("right_angle_candidates_eval_time_total", -1.0)) == 0.0
    assert float(metrics.get("right_angle_extra_overhead_time", -1.0)) == 0.0


