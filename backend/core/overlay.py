"""
疊圖生成模組
用於生成印章疊圖視覺化
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def create_overlay_image(
    image1_path: str,
    image2_path: str,
    overlay_dir: Path,
    record_id: str,
    image2_corrected_path: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    創建疊圖比對圖像
    
    Args:
        image1_path: 第一個圖像路徑
        image2_path: 第二個圖像路徑（原始）
        overlay_dir: 疊圖輸出目錄
        record_id: 記錄 ID（用於檔案命名）
        image2_corrected_path: 校正後的圖像2路徑（如果存在，優先使用）
        
    Returns:
        (overlay1_url, overlay2_url) - 兩個疊圖的相對路徑，失敗返回 (None, None)
    """
    # 優先使用校正後的圖像2
    if image2_corrected_path:
        image2_path = image2_corrected_path
    
    try:
        # 轉換路徑（處理容器路徑）
        def normalize_path(p):
            if not p:
                return None
            s = str(p)
            if s.startswith('/app/'):
                s = s.replace('/app/', '')
            return Path(s)
        
        img1_path = normalize_path(image1_path)
        img2_path = normalize_path(image2_path)
        
        if not img1_path or not img2_path:
            return None, None
        
        # 檢查檔案是否存在
        if not img1_path.exists() or not img2_path.exists():
            return None, None
        
        # 讀取圖像
        img1 = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
        img2 = cv2.imread(str(img2_path), cv2.IMREAD_COLOR)
        
        if img1 is None or img2 is None:
            return None, None
        
        # 調整到相同尺寸（使用高質量插值，與並排比對截圖保持一致）
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        target_h = max(h1, h2)
        target_w = max(w1, w2)
        
        # 使用 INTER_LINEAR 插值以保持高質量（與 verification.py 保持一致）
        img1_resized = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img2_resized = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # 背景移除和透明化處理
        def remove_background_and_make_transparent(img):
            """
            移除背景並創建透明圖像
            使用多種方法檢測背景，優先考慮圖像邊緣的顏色
            
            Args:
                img: BGR 圖像
                
            Returns:
                (mask, rgba_image) - 印章遮罩和帶透明通道的圖像
            """
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 方法1: 檢測圖像邊緣的顏色（通常邊緣是背景）
            edge_width = max(5, min(h, w) // 20)
            
            # 收集邊緣像素
            edge_pixels = []
            edge_pixels.extend(img[0:edge_width, :].reshape(-1, 3).tolist())
            edge_pixels.extend(img[h-edge_width:h, :].reshape(-1, 3).tolist())
            edge_pixels.extend(img[:, 0:edge_width].reshape(-1, 3).tolist())
            edge_pixels.extend(img[:, w-edge_width:w].reshape(-1, 3).tolist())
            
            edge_colors = np.array(edge_pixels, dtype=np.float32)
            bg_color = np.median(edge_colors, axis=0).astype(np.uint8)
            
            # 計算每個像素與背景顏色的距離
            img_float = img.astype(np.float32)
            bg_float = bg_color.astype(np.float32)
            color_diff = np.sqrt(np.sum((img_float - bg_float) ** 2, axis=2))
            
            # 設定閾值
            edge_std = np.std(edge_colors, axis=0).mean()
            threshold = max(25, min(50, edge_std * 2))
            bg_mask = color_diff < threshold
            
            # 方法2: 根據亮度調整
            mean_brightness = np.mean(gray)
            
            if mean_brightness > 200:
                _, bright_bg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
                bright_bg_mask = bright_bg_mask > 0
                bg_mask = bg_mask | bright_bg_mask
            elif mean_brightness < 50:
                _, dark_bg_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
                dark_bg_mask = dark_bg_mask > 0
                bg_mask = bg_mask | dark_bg_mask
            
            # 印章遮罩是背景的反轉
            seal_mask = (~bg_mask).astype(np.uint8) * 255
            
            # 形態學操作優化遮罩
            kernel = np.ones((3, 3), np.uint8)
            seal_mask = cv2.morphologyEx(seal_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            seal_mask = cv2.morphologyEx(seal_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 如果印章區域太小，使用 OTSU 作為備選
            seal_area = np.sum(seal_mask > 0)
            total_area = h * w
            if seal_area < total_area * 0.01:
                _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                seal_mask = binary_otsu
                seal_mask = cv2.morphologyEx(seal_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                seal_mask = cv2.morphologyEx(seal_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 確保印章 mask 是二值化的
            _, seal_mask = cv2.threshold(seal_mask, 127, 255, cv2.THRESH_BINARY)
            
            # 創建 RGBA 圖像
            rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = seal_mask
            
            return seal_mask, rgba
        
        # 處理兩個圖像
        mask1, img1_rgba = remove_background_and_make_transparent(img1_resized)
        mask2, img2_rgba = remove_background_and_make_transparent(img2_resized)
        
        # 轉換為灰度圖以便比對
        gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY) if len(img1_resized.shape) == 3 else img1_resized
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY) if len(img2_resized.shape) == 3 else img2_resized
        
        # 使用 mask 來獲取印章部分的二值化圖像
        binary1 = np.zeros_like(gray1)
        binary1[mask1 > 0] = 255
        
        binary2 = np.zeros_like(gray2)
        binary2[mask2 > 0] = 255
        
        # 創建彩色疊圖（OpenCV 使用 BGR 格式）
        # 圖像1用藍色 [255, 0, 0] (BGR)，圖像2用紅色 [0, 0, 255] (BGR)
        overlay1_on_2 = np.zeros((target_h, target_w, 3), dtype=np.uint8)  # 圖像1疊在圖像2上
        overlay2_on_1 = np.zeros((target_h, target_w, 3), dtype=np.uint8)  # 圖像2疊在圖像1上
        
        # 計算差異區域
        diff_mask_2_only = (binary2 > 0) & (binary1 == 0)  # 只有圖像2有
        diff_mask_1_only = (binary1 > 0) & (binary2 == 0)  # 只有圖像1有
        
        # 疊圖1：圖像1（藍色）疊在圖像2（紅色）上，顯示圖像2多出的部分（黃色）
        overlay1_on_2[binary2 > 0] = [0, 0, 255]  # 紅色（圖像2，BGR格式）
        overlay1_on_2[binary1 > 0] = [255, 0, 0]  # 藍色（圖像1，BGR格式）
        overlay1_on_2[diff_mask_2_only] = [0, 255, 255]  # 黃色（圖像2多出部分，BGR格式）
        
        # 疊圖2：圖像2（紅色）疊在圖像1（藍色）上，顯示圖像1多出的部分（黃色）
        overlay2_on_1[binary1 > 0] = [255, 0, 0]  # 藍色（圖像1，BGR格式）
        overlay2_on_1[binary2 > 0] = [0, 0, 255]  # 紅色（圖像2，BGR格式）
        overlay2_on_1[diff_mask_1_only] = [0, 255, 255]  # 黃色（圖像1多出部分，BGR格式）
        
        # 創建帶透明背景的疊圖（使用 PNG 格式支持透明度）
        overlay1_rgba = np.zeros((target_h, target_w, 4), dtype=np.uint8)
        overlay2_rgba = np.zeros((target_h, target_w, 4), dtype=np.uint8)
        
        # 疊圖1
        overlay1_rgba[:, :, :3] = overlay1_on_2
        overlay1_rgba[:, :, 3] = np.maximum(binary1, binary2)
        
        # 疊圖2
        overlay2_rgba[:, :, :3] = overlay2_on_1
        overlay2_rgba[:, :, 3] = np.maximum(binary1, binary2)
        
        # 保存疊圖（使用高質量 PNG 壓縮，確保解析度與並排比對截圖一致）
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay1_file = overlay_dir / f"overlay_{record_id}_img1_on_img2.png"
        overlay2_file = overlay_dir / f"overlay_{record_id}_img2_on_img1.png"
        
        # 使用 PNG 壓縮級別 1（最高質量，接近無損）以確保解析度一致
        # cv2.IMWRITE_PNG_COMPRESSION 的範圍是 0-9，0 是無損，1 是最高質量
        cv2.imwrite(str(overlay1_file), overlay1_rgba, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(str(overlay2_file), overlay2_rgba, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        
        # 返回絕對路徑（用於資料庫存儲）
        return str(overlay1_file), str(overlay2_file)
        
    except Exception as e:
        print(f"警告：無法生成疊圖 {record_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

