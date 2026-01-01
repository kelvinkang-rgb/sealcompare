import cv2
import numpy as np

from core.overlay import calculate_structure_similarity, calculate_structure_similarity_from_images


def test_structure_similarity_inmemory_matches_path(tmp_path):
    # 合成一張簡單的非對稱印樣
    img = np.full((180, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (20, 30), (70, 140), (0, 0, 0), -1)
    cv2.circle(img, (140, 60), 12, (0, 0, 0), -1)
    cv2.line(img, (110, 150), (190, 150), (0, 0, 0), 6)

    # 做一個「深淺變化版本」，結構應一致
    img2 = img.copy()
    # 讓黑色筆劃變淡一點（仍保留邊緣）
    mask = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) < 250
    img2[mask] = np.clip(img2[mask].astype(np.int16) + 80, 0, 255).astype(np.uint8)

    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    cv2.imwrite(str(p1), img)
    cv2.imwrite(str(p2), img2)

    sim_path = calculate_structure_similarity(str(p1), str(p2))
    sim_mem = calculate_structure_similarity_from_images(img, img2)

    assert abs(float(sim_path) - float(sim_mem)) < 1e-9


