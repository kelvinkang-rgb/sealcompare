"""
生成測試用的印章圖像
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_seal_image(text: str, size: int = 500, color: tuple = (0, 0, 255)) -> np.ndarray:
    """
    創建一個正方形印章圖像
    
    Args:
        text: 印章文字
        size: 圖像尺寸
        color: 印章顏色 (BGR格式)
        
    Returns:
        印章圖像
    """
    # 創建白色背景 (使用 PIL 以便支援中文)
    img_pil = Image.new('RGB', (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img_pil)
    
    # 轉換顏色格式 (BGR -> RGB)
    rgb_color = (color[2], color[1], color[0])
    
    # 繪製正方形邊框
    margin = 30
    # 外框
    draw.rectangle([margin, margin, size - margin, size - margin], 
                   outline=rgb_color, width=4)
    # 內框
    inner_margin = margin + 15
    draw.rectangle([inner_margin, inner_margin, size - inner_margin, size - inner_margin], 
                   outline=rgb_color, width=2)
    
    # 繪製中文文字
    font_size = int(size * 0.25)
    font = None
    # 嘗試多種中文字體路徑
    font_paths = [
        "msjh.ttc",  # Windows 微軟正黑體
        "C:/Windows/Fonts/msjh.ttc",  # Windows 完整路徑
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux 文泉驛正黑
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux 文泉驛微米黑
        "/System/Library/Fonts/PingFang.ttc",  # macOS 蘋方
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except:
            continue
    
    # 如果所有字體都找不到，使用默認字體
    if font is None:
        font = ImageFont.load_default()
    
    # 計算文字位置（居中）
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (size - text_width) // 2
    text_y = (size - text_height) // 2
    
    # 繪製文字
    draw.text((text_x, text_y), text, fill=rgb_color, font=font)
    
    # 添加一些裝飾性圖案（在四個角落添加小圓點）
    corner_offset = 50
    dot_size = 6
    draw.ellipse([corner_offset - dot_size, corner_offset - dot_size,
                  corner_offset + dot_size, corner_offset + dot_size], 
                 fill=rgb_color)
    draw.ellipse([size - corner_offset - dot_size, corner_offset - dot_size,
                  size - corner_offset + dot_size, corner_offset + dot_size], 
                 fill=rgb_color)
    draw.ellipse([corner_offset - dot_size, size - corner_offset - dot_size,
                  corner_offset + dot_size, size - corner_offset + dot_size], 
                 fill=rgb_color)
    draw.ellipse([size - corner_offset - dot_size, size - corner_offset - dot_size,
                  size - corner_offset + dot_size, size - corner_offset + dot_size], 
                 fill=rgb_color)
    
    # 轉換回 OpenCV 格式 (RGB -> BGR)
    img = np.array(img_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img


def create_similar_seal(text: str, size: int = 500, noise_level: float = 0.02) -> np.ndarray:
    """
    創建一個相似的印章圖像（添加少量噪聲）
    
    Args:
        text: 印章文字
        size: 圖像尺寸
        noise_level: 噪聲水平
        
    Returns:
        相似的印章圖像
    """
    img = create_seal_image(text, size)
    
    # 添加輕微噪聲
    noise = np.random.randint(0, int(255 * noise_level), img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    
    return img


def create_rotated_seal(text: str, size: int = 500, angle: float = 15.0) -> np.ndarray:
    """
    創建一個旋轉的印章圖像（模擬蓋歪了的情況）
    
    Args:
        text: 印章文字
        size: 圖像尺寸
        angle: 旋轉角度（度數）
        
    Returns:
        旋轉後的印章圖像
    """
    # 創建一個更大的畫布以容納旋轉後的圖像
    canvas_size = int(size * 1.5)
    img = create_seal_image(text, size)
    
    # 轉換為 PIL 以便旋轉
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 旋轉圖像
    img_rotated = img_pil.rotate(angle, expand=False, fillcolor=(255, 255, 255))
    
    # 轉換回 OpenCV 格式
    img_rotated_cv = cv2.cvtColor(np.array(img_rotated), cv2.COLOR_RGB2BGR)
    
    # 裁剪回原始尺寸（居中）
    h, w = img_rotated_cv.shape[:2]
    start_y = (h - size) // 2
    start_x = (w - size) // 2
    img_cropped = img_rotated_cv[start_y:start_y+size, start_x:start_x+size]
    
    return img_cropped


def create_tilted_seal(text: str, size: int = 500, tilt_angle: float = 10.0) -> np.ndarray:
    """
    創建一個傾斜的印章圖像（模擬不正的情況，使用仿射變換）
    
    Args:
        text: 印章文字
        size: 圖像尺寸
        tilt_angle: 傾斜角度（度數）
        
    Returns:
        傾斜後的印章圖像
    """
    img = create_seal_image(text, size)
    
    # 計算傾斜變換矩陣
    tilt_rad = np.radians(tilt_angle)
    # 水平傾斜（shear）
    M = np.float32([
        [1, np.tan(tilt_rad), 0],
        [0, 1, 0]
    ])
    
    # 應用仿射變換
    img_tilted = cv2.warpAffine(img, M, (size, size), 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(255, 255, 255))
    
    return img_tilted


def create_scaled_seal(text: str, size: int = 500, scale: float = 0.9) -> np.ndarray:
    """
    創建一個縮放的印章圖像（模擬蓋印時壓力不同）
    
    Args:
        text: 印章文字
        size: 圖像尺寸
        scale: 縮放比例（小於1表示縮小，大於1表示放大，但會裁剪）
        
    Returns:
        縮放後的印章圖像
    """
    img = create_seal_image(text, size)
    
    # 計算新尺寸
    new_size = int(size * scale)
    
    # 如果放大，先創建更大的圖像，然後裁剪
    if scale > 1.0:
        # 創建更大的畫布
        canvas_size = int(size * scale * 1.2)  # 留一些邊距
        img_large = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        
        # 將原圖放在中心
        start_y = (canvas_size - size) // 2
        start_x = (canvas_size - size) // 2
        img_large[start_y:start_y+size, start_x:start_x+size] = img
        
        # 縮放到新尺寸
        img_scaled = cv2.resize(img_large, (new_size, new_size), interpolation=cv2.INTER_AREA)
        
        # 裁剪回原始尺寸（居中）
        h, w = img_scaled.shape[:2]
        start_y = (h - size) // 2
        start_x = (w - size) // 2
        img_result = img_scaled[start_y:start_y+size, start_x:start_x+size]
    else:
        # 縮放圖像
        img_scaled = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_AREA)
        
        # 創建白色背景並居中放置
        img_result = np.ones((size, size, 3), dtype=np.uint8) * 255
        start_y = (size - new_size) // 2
        start_x = (size - new_size) // 2
        img_result[start_y:start_y+new_size, start_x:start_x+new_size] = img_scaled
    
    return img_result


def create_translated_seal(text: str, size: int = 500, offset_x: int = 20, offset_y: int = 20) -> np.ndarray:
    """
    創建一個平移的印章圖像（模擬蓋印位置偏移）
    
    Args:
        text: 印章文字
        size: 圖像尺寸
        offset_x: X方向偏移（像素）
        offset_y: Y方向偏移（像素）
        
    Returns:
        平移後的印章圖像
    """
    img = create_seal_image(text, size)
    
    # 創建平移矩陣
    M = np.float32([
        [1, 0, offset_x],
        [0, 1, offset_y]
    ])
    
    # 應用平移變換
    img_translated = cv2.warpAffine(img, M, (size, size),
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))
    
    return img_translated


def create_perspective_seal(text: str, size: int = 500, perspective_strength: float = 0.05) -> np.ndarray:
    """
    創建一個透視變形的印章圖像（模擬從側面拍攝或蓋印時角度不正）
    
    Args:
        text: 印章文字
        size: 圖像尺寸
        perspective_strength: 透視變形強度（0-1）
        
    Returns:
        透視變形後的印章圖像
    """
    img = create_seal_image(text, size)
    
    # 定義原始四個角點
    pts1 = np.float32([
        [0, 0],
        [size, 0],
        [size, size],
        [0, size]
    ])
    
    # 定義變形後的四個角點（輕微透視變形）
    offset = int(size * perspective_strength)
    pts2 = np.float32([
        [offset, 0],
        [size - offset, 0],
        [size, size],
        [0, size]
    ])
    
    # 計算透視變換矩陣
    M = cv2.getPerspectiveTransform(pts1, pts2)
    
    # 應用透視變換
    img_perspective = cv2.warpPerspective(img, M, (size, size),
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=(255, 255, 255))
    
    return img_perspective


def create_different_seal(text: str, size: int = 500) -> np.ndarray:
    """
    創建一個不同的印章圖像（使用不同的樣式）
    
    Args:
        text: 印章文字
        size: 圖像尺寸
        
    Returns:
        不同的印章圖像
    """
    # 創建白色背景 (使用 PIL 以便支援中文)
    img_pil = Image.new('RGB', (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img_pil)
    
    # 使用不同的顏色（藍色）
    color = (255, 0, 0)  # RGB 格式的藍色
    rgb_color = color
    
    # 繪製正方形邊框（使用不同的樣式）
    margin = 50
    # 外框
    draw.rectangle([margin, margin, size - margin, size - margin], 
                   outline=rgb_color, width=5)
    # 內框
    inner_margin = margin + 20
    draw.rectangle([inner_margin, inner_margin, size - inner_margin, size - inner_margin], 
                   outline=rgb_color, width=3)
    
    # 繪製中文文字
    font_size = int(size * 0.25)
    font = None
    # 嘗試多種中文字體路徑
    font_paths = [
        "msjh.ttc",  # Windows 微軟正黑體
        "C:/Windows/Fonts/msjh.ttc",  # Windows 完整路徑
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux 文泉驛正黑
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux 文泉驛微米黑
        "/System/Library/Fonts/PingFang.ttc",  # macOS 蘋方
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except:
            continue
    
    # 如果所有字體都找不到，使用默認字體
    if font is None:
        font = ImageFont.load_default()
    
    # 計算文字位置（居中）
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (size - text_width) // 2
    text_y = (size - text_height) // 2
    
    # 繪製文字
    draw.text((text_x, text_y), text, fill=rgb_color, font=font)
    
    # 添加不同的裝飾（在中心添加星形）
    center = (size // 2, size // 2)
    star_size = 40
    points = []
    for i in range(5):
        angle = i * 2 * np.pi / 5 - np.pi / 2
        x = int(center[0] + star_size * np.cos(angle))
        y = int(center[1] + star_size * np.sin(angle))
        points.append((x, y))
    
    # 繪製星形
    draw.polygon(points, fill=rgb_color)
    
    # 轉換回 OpenCV 格式 (RGB -> BGR)
    img = np.array(img_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img


def main():
    """生成測試圖像"""
    # 創建測試圖像目錄
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    print("正在生成測試圖像...")
    
    # 1. 兩個完全相同的印章（應該比對成功）
    seal1 = create_seal_image("王大明", size=500)
    seal2 = create_seal_image("王大明", size=500)
    cv2.imwrite(str(test_dir / "seal_original_1.jpg"), seal1)
    cv2.imwrite(str(test_dir / "seal_original_2.jpg"), seal2)
    print("✓ 已創建: seal_original_1.jpg 和 seal_original_2.jpg (完全相同)")
    
    # 2. 一個相似的印章（添加輕微噪聲，應該比對成功但相似度略低）
    seal3 = create_similar_seal("王大明", size=500, noise_level=0.01)
    cv2.imwrite(str(test_dir / "seal_similar.jpg"), seal3)
    print("✓ 已創建: seal_similar.jpg (相似但略有差異)")
    
    # 3. 一個不同的印章（應該比對失敗）
    seal4 = create_different_seal("李小明", size=500)
    cv2.imwrite(str(test_dir / "seal_different.jpg"), seal4)
    print("✓ 已創建: seal_different.jpg (完全不同的印章)")
    
    # 4. 另一個不同的印章（用於測試兩個不同印章的比對）
    seal5 = create_seal_image("陳小華", size=500, color=(0, 128, 0))
    cv2.imwrite(str(test_dir / "seal_company.jpg"), seal5)
    print("✓ 已創建: seal_company.jpg (另一個不同的印章)")
    
    # 5. 創建一個稍微不同的版本（用於測試邊界情況）
    seal6 = create_similar_seal("王大明", size=500, noise_level=0.05)
    cv2.imwrite(str(test_dir / "seal_noisy.jpg"), seal6)
    print("✓ 已創建: seal_noisy.jpg (有較多噪聲的版本)")
    
    # 6. 創建旋轉的印章（蓋歪了 - 小角度）
    seal7 = create_rotated_seal("王大明", size=500, angle=8.0)
    cv2.imwrite(str(test_dir / "seal_rotated_small.jpg"), seal7)
    print("✓ 已創建: seal_rotated_small.jpg (輕微旋轉 8度)")
    
    # 7. 創建旋轉的印章（蓋歪了 - 中等角度）
    seal8 = create_rotated_seal("王大明", size=500, angle=15.0)
    cv2.imwrite(str(test_dir / "seal_rotated_medium.jpg"), seal8)
    print("✓ 已創建: seal_rotated_medium.jpg (中等旋轉 15度)")
    
    # 8. 創建旋轉的印章（蓋歪了 - 大角度）
    seal9 = create_rotated_seal("王大明", size=500, angle=25.0)
    cv2.imwrite(str(test_dir / "seal_rotated_large.jpg"), seal9)
    print("✓ 已創建: seal_rotated_large.jpg (大幅旋轉 25度)")
    
    # 9. 創建傾斜的印章（不正 - 輕微）
    seal10 = create_tilted_seal("王大明", size=500, tilt_angle=5.0)
    cv2.imwrite(str(test_dir / "seal_tilted_small.jpg"), seal10)
    print("✓ 已創建: seal_tilted_small.jpg (輕微傾斜 5度)")
    
    # 10. 創建傾斜的印章（不正 - 中等）
    seal11 = create_tilted_seal("王大明", size=500, tilt_angle=12.0)
    cv2.imwrite(str(test_dir / "seal_tilted_medium.jpg"), seal11)
    print("✓ 已創建: seal_tilted_medium.jpg (中等傾斜 12度)")
    
    # 11. 創建縮小的印章（蓋印壓力小）
    seal12 = create_scaled_seal("王大明", size=500, scale=0.85)
    cv2.imwrite(str(test_dir / "seal_scaled_small.jpg"), seal12)
    print("✓ 已創建: seal_scaled_small.jpg (縮小 85%)")
    
    # 12. 創建放大的印章（蓋印壓力大）
    seal13 = create_scaled_seal("王大明", size=500, scale=1.1)
    cv2.imwrite(str(test_dir / "seal_scaled_large.jpg"), seal13)
    print("✓ 已創建: seal_scaled_large.jpg (放大 110%)")
    
    # 13. 創建平移的印章（位置偏移）
    seal14 = create_translated_seal("王大明", size=500, offset_x=15, offset_y=15)
    cv2.imwrite(str(test_dir / "seal_translated.jpg"), seal14)
    print("✓ 已創建: seal_translated.jpg (位置偏移)")
    
    # 14. 創建透視變形的印章（角度不正）
    seal15 = create_perspective_seal("王大明", size=500, perspective_strength=0.08)
    cv2.imwrite(str(test_dir / "seal_perspective.jpg"), seal15)
    print("✓ 已創建: seal_perspective.jpg (透視變形)")
    
    # 15. 創建組合變形（旋轉 + 傾斜）
    seal16 = create_rotated_seal("王大明", size=500, angle=10.0)
    seal16_pil = Image.fromarray(cv2.cvtColor(seal16, cv2.COLOR_BGR2RGB))
    seal16_tilted = seal16_pil.transform(
        (500, 500),
        Image.AFFINE,
        [1, np.tan(np.radians(8)), 0, 0, 1, 0],
        fillcolor=(255, 255, 255)
    )
    seal16_final = cv2.cvtColor(np.array(seal16_tilted), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(test_dir / "seal_combined.jpg"), seal16_final)
    print("✓ 已創建: seal_combined.jpg (旋轉+傾斜組合)")
    
    print(f"\n所有測試圖像已生成在 '{test_dir}' 目錄中")
    print("\n測試建議：")
    print("  1. 比對相同印章：")
    print("     python main.py test_images/seal_original_1.jpg test_images/seal_original_2.jpg")
    print("  2. 比對相似印章（輕微噪聲）：")
    print("     python main.py test_images/seal_original_1.jpg test_images/seal_similar.jpg")
    print("  3. 比對不同印章：")
    print("     python main.py test_images/seal_original_1.jpg test_images/seal_different.jpg")
    print("  4. 測試旋轉（蓋歪了）：")
    print("     python main.py test_images/seal_original_1.jpg test_images/seal_rotated_small.jpg")
    print("     python main.py test_images/seal_original_1.jpg test_images/seal_rotated_medium.jpg")
    print("     python main.py test_images/seal_original_1.jpg test_images/seal_rotated_large.jpg")
    print("  5. 測試傾斜（不正）：")
    print("     python main.py test_images/seal_original_1.jpg test_images/seal_tilted_small.jpg")
    print("     python main.py test_images/seal_original_1.jpg test_images/seal_tilted_medium.jpg")
    print("  6. 測試縮放（壓力不同）：")
    print("     python main.py test_images/seal_original_1.jpg test_images/seal_scaled_small.jpg")
    print("     python main.py test_images/seal_original_1.jpg test_images/seal_scaled_large.jpg")
    print("  7. 測試位置偏移：")
    print("     python main.py test_images/seal_original_1.jpg test_images/seal_translated.jpg")
    print("  8. 測試透視變形：")
    print("     python main.py test_images/seal_original_1.jpg test_images/seal_perspective.jpg")
    print("  9. 測試組合變形：")
    print("     python main.py test_images/seal_original_1.jpg test_images/seal_combined.jpg")


if __name__ == '__main__':
    main()

