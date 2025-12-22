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
    
    生成兩張疊圖：
    1. 疊圖1：圖像1疊在圖像2校正上
    2. 疊圖2：圖像2校正疊在圖像1上
    
    顯示規則：
    - 如果兩個圖像尺寸不同，會裁剪到較小圖像的尺寸（從左上角開始），只比較重疊區域
    - 重疊區域且沒有差異的部分：使用原始圖像顏色呈現（疊圖1使用圖像1顏色，疊圖2使用圖像2顏色）
    - 差異區域（包括圖像1獨有區域、圖像2獨有區域、重疊區域內的像素差異）：使用黑色標記
    - 背景區域：透明（PNG 格式支持透明度）
    
    Args:
        image1_path: 第一個圖像路徑
        image2_path: 第二個圖像路徑（原始）
        overlay_dir: 疊圖輸出目錄
        record_id: 記錄 ID（用於檔案命名）
        image2_corrected_path: 校正後的圖像2路徑（如果存在，優先使用）
        
    Returns:
        (overlay1_url, overlay2_url) - 兩個疊圖的絕對路徑，失敗返回 (None, None)
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
        
        # 獲取圖像尺寸
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 裁剪到較小圖像的尺寸（只比較重疊區域）
        min_h = min(h1, h2)
        min_w = min(w1, w2)
        
        # 從左上角開始裁剪
        img1_cropped = img1[0:min_h, 0:min_w]
        img2_cropped = img2[0:min_h, 0:min_w]
        
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
        mask1, img1_rgba = remove_background_and_make_transparent(img1_cropped)
        mask2, img2_rgba = remove_background_and_make_transparent(img2_cropped)
        
        # 轉換為灰度圖以便比對
        gray1 = cv2.cvtColor(img1_cropped, cv2.COLOR_BGR2GRAY) if len(img1_cropped.shape) == 3 else img1_cropped
        gray2 = cv2.cvtColor(img2_cropped, cv2.COLOR_BGR2GRAY) if len(img2_cropped.shape) == 3 else img2_cropped
        
        # 使用 mask 來獲取印章部分的二值化圖像
        binary1 = np.zeros_like(gray1)
        binary1[mask1 > 0] = 255
        
        binary2 = np.zeros_like(gray2)
        binary2[mask2 > 0] = 255
        
        # 差異區域使用黑色 (BGR格式: [0, 0, 0])
        diff_color = np.array([0, 0, 0], dtype=np.uint8)  # 黑色
        
        # 計算差異區域
        diff_mask_2_only = (binary2 > 0) & (binary1 == 0)  # 只有圖像2有
        diff_mask_1_only = (binary1 > 0) & (binary2 == 0)  # 只有圖像1有
        overlap_mask = (binary1 > 0) & (binary2 > 0)  # 兩者重疊區域
        
        # 計算像素差異（在重疊區域內）
        diff_threshold = 30  # 差異閾值
        gray_diff = cv2.absdiff(gray1, gray2)
        pixel_diff_mask = (gray_diff > diff_threshold) & overlap_mask  # 重疊區域內的像素差異
        
        # 所有差異區域（包括獨有區域和重疊區域內的像素差異）
        all_diff_mask = diff_mask_1_only | diff_mask_2_only | pixel_diff_mask
        
        # 創建彩色疊圖（OpenCV 使用 BGR 格式）
        # 重疊區域用原色呈現，差異區域用黑色標記
        overlay1_on_2 = np.zeros((min_h, min_w, 3), dtype=np.uint8)  # 圖像1疊在圖像2校正上
        overlay2_on_1 = np.zeros((min_h, min_w, 3), dtype=np.uint8)  # 圖像2校正疊在圖像1上
        
        # 計算重疊區域且沒有差異的部分
        overlap_no_diff = overlap_mask & (~pixel_diff_mask)
        
        # 疊圖1：圖像1疊在圖像2校正上
        # 設置重疊區域且沒有差異的部分：使用圖像1的原始顏色（因為圖像1在上層）
        overlay1_on_2[overlap_no_diff] = img1_cropped[overlap_no_diff]
        # 標記所有差異區域（使用黑色）
        # 差異區域包括：圖像1獨有區域、圖像2校正獨有區域、重疊區域內的像素差異
        overlay1_on_2[all_diff_mask] = diff_color
        
        # 疊圖2：圖像2校正疊在圖像1上
        # 設置重疊區域且沒有差異的部分：使用圖像2校正的原始顏色（因為圖像2校正在上層）
        overlay2_on_1[overlap_no_diff] = img2_cropped[overlap_no_diff]
        # 標記所有差異區域（使用黑色）
        overlay2_on_1[all_diff_mask] = diff_color
        
        # 創建帶透明背景的疊圖（使用 PNG 格式支持透明度）
        overlay1_rgba = np.zeros((min_h, min_w, 4), dtype=np.uint8)
        overlay2_rgba = np.zeros((min_h, min_w, 4), dtype=np.uint8)
        
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

