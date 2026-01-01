import cv2
import numpy as np

from core.seal_compare import SealComparator


def _make_asymmetric_stamp(size: int = 220) -> np.ndarray:
    """
    產生一張非對稱的灰階「印樣」測試圖，避免 90/180/270 旋轉下出現對稱歧義。
    背景白(255)，筆劃黑(0)。
    """
    img = np.full((size, size), 255, dtype=np.uint8)

    # L 形主結構
    cv2.line(img, (40, 30), (40, size - 30), 0, 12)
    cv2.line(img, (40, size - 35), (size - 55, size - 35), 0, 12)

    # 加一個偏心圓點打破 180/270 的潛在對稱
    cv2.circle(img, (size - 65, 70), 10, 0, -1)

    # 再加一個短橫，增加方向辨識訊號
    cv2.line(img, (85, 55), (135, 55), 0, 8)

    return img


def _rotate_ccw(img: np.ndarray, degrees: int) -> np.ndarray:
    degrees = degrees % 360
    if degrees == 0:
        return img.copy()
    if degrees == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if degrees == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    raise ValueError(f"Only supports right-angle rotations; got {degrees}")


def test_align_right_angle_rotation_fallback_picks_correct_base():
    comp = SealComparator()
    img1 = _make_asymmetric_stamp()

    # img2 是 img1 的右角旋轉版本；要對齊回 img1，需要套用相反方向的 base 旋轉：
    # expected_base = (360 - rot) % 360
    cases = [
        (90, 270),
        (180, 180),
        (270, 90),
    ]

    for rot, expected_base in cases:
        img2 = _rotate_ccw(img1, rot)
        aligned, angle, (ox, oy), sim, metrics, timing = comp._align_image2_to_image1(
            img1, img2, rotation_range=15.0, translation_range=100
        )

        assert isinstance(metrics, dict)
        assert metrics.get("right_angle_fallback_used") is True
        assert metrics.get("right_angle_base_rotation") == expected_base
        assert metrics.get("right_angle_candidate_mode") == "stage45_only"

        # 這個測試圖是同一張圖右角旋轉後再對齊，理論上應該能得到很高相似度
        assert float(sim) >= 0.85

        # 基本 sanity：輸出應為有效影像
        assert aligned is not None and aligned.size > 0


