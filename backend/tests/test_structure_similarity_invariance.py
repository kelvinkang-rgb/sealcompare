import cv2
import numpy as np

from core.overlay import calculate_structure_similarity


def _draw_asymmetric_seal(size: int = 256, thickness: int = 6) -> np.ndarray:
    """白底黑線的非對稱幾何，降低「不同圖也意外高分」的機率。"""
    img = np.ones((size, size, 3), dtype=np.uint8) * 255

    # outer ring (slightly off-center)
    cv2.circle(img, (size // 2 + 5, size // 2 - 3), size // 3, (0, 0, 0), thickness)
    # inner motif: L-shape + diagonal
    cv2.line(img, (60, 70), (60, 180), (0, 0, 0), thickness)
    cv2.line(img, (60, 180), (170, 180), (0, 0, 0), thickness)
    cv2.line(img, (90, 80), (180, 140), (0, 0, 0), thickness)
    # a small dot (asymmetry)
    cv2.circle(img, (185, 85), 8, (0, 0, 0), -1)

    return img


def _ink_variation(img: np.ndarray, k: float) -> np.ndarray:
    """
    模擬印泥深淺：把「黑色筆畫」往白色拉（k 越小越淡），背景仍接近白。
    new = 255 - (255 - img) * k
    """
    img_f = img.astype(np.float32)
    out = 255.0 - (255.0 - img_f) * float(k)
    return np.clip(out, 0, 255).astype(np.uint8)


def test_structure_similarity_is_invariant_to_ink_depth(tmp_path):
    base = _draw_asymmetric_seal()
    # lighter / darker ink
    lighter = _ink_variation(base, k=0.45)
    darker = _ink_variation(base, k=0.90)

    p_base = str(tmp_path / "base.png")
    p_lighter = str(tmp_path / "lighter.png")
    p_darker = str(tmp_path / "darker.png")
    cv2.imwrite(p_base, base)
    cv2.imwrite(p_lighter, lighter)
    cv2.imwrite(p_darker, darker)

    sim_light = calculate_structure_similarity(p_base, p_lighter)
    sim_dark = calculate_structure_similarity(p_base, p_darker)

    # 同章深淺變化：分數應維持高（避免吃灰階強度）
    assert sim_light > 0.65
    assert sim_dark > 0.75


def test_structure_similarity_rejects_different_geometry(tmp_path):
    base = _draw_asymmetric_seal()
    different = _draw_asymmetric_seal()
    # remove one stroke to simulate different seal
    cv2.line(different, (60, 180), (170, 180), (255, 255, 255), 14)

    p_base = str(tmp_path / "base.png")
    p_diff = str(tmp_path / "diff.png")
    p_var = str(tmp_path / "var.png")
    cv2.imwrite(p_base, base)
    cv2.imwrite(p_diff, different)
    cv2.imwrite(p_var, _ink_variation(base, k=0.65))

    sim_same = calculate_structure_similarity(p_base, p_var)
    sim_diff = calculate_structure_similarity(p_base, p_diff)

    # 避免假陽性：不同幾何要明顯更低
    assert sim_same > sim_diff + 0.15
    assert sim_diff < 0.75


