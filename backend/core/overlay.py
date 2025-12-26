"""
疊圖生成模組
用於生成印章疊圖視覺化
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any


def create_overlay_image(
    image1_path: str,
    image2_path: str,
    overlay_dir: Path,
    record_id: str,
    image2_corrected_path: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    創建疊圖比對圖像
    
    生成兩張疊圖、四個mask圖像和一個灰度差異圖：
    1. 疊圖1：圖像1疊在圖像2校正上
    2. 疊圖2：圖像2校正疊在圖像1上
    3. overlap_mask：重疊區域mask（二值圖像：白色=重疊，黑色=非重疊）
    4. pixel_diff_mask：重疊區域內的像素差異mask（二值圖像：白色=差異，黑色=無差異）
    5. diff_mask_2_only：只有圖像2有的區域mask（二值圖像：白色=圖像2獨有，黑色=其他）
    6. diff_mask_1_only：只有圖像1有的區域mask（二值圖像：白色=圖像1獨有，黑色=其他）
    7. gray_diff：灰度差異圖（熱力圖：顏色越熱表示差異越大，0=無差異，255=最大差異）
    
    顯示規則：
    - 如果兩個圖像尺寸不同，會調整到較大圖像的尺寸（較小的圖像用白色背景填充到左上角）
    - 重疊區域且沒有差異的部分：使用原始圖像顏色呈現（疊圖1使用圖像1顏色，疊圖2使用圖像2顏色）
    - 差異區域（包括圖像1獨有區域、圖像2獨有區域、重疊區域內的像素差異）：使用黑色或亮橘色標記
    - 背景區域：透明（PNG 格式支持透明度）
    
    Args:
        image1_path: 第一個圖像路徑
        image2_path: 第二個圖像路徑（原始）
        overlay_dir: 疊圖輸出目錄
        record_id: 記錄 ID（用於檔案命名）
        image2_corrected_path: 校正後的圖像2路徑（如果存在，優先使用）
        
    Returns:
        (overlay1_path, overlay2_path, overlap_mask_path, pixel_diff_mask_path, diff_mask_2_only_path, diff_mask_1_only_path, gray_diff_path)
        - 7個圖像的絕對路徑，失敗返回 (None, None, None, None, None, None, None)
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
            return None, None, None, None, None, None, None
        
        # 檢查檔案是否存在
        if not img1_path.exists() or not img2_path.exists():
            return None, None, None, None, None, None, None
        
        # 讀取圖像
        img1 = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
        img2 = cv2.imread(str(img2_path), cv2.IMREAD_COLOR)
        
        if img1 is None or img2 is None:
            return None, None, None, None, None, None, None
        
        # 獲取圖像尺寸
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 使用較大圖像的尺寸
        max_h = max(h1, h2)
        max_w = max(w1, w2)
        
        # 將兩個圖像調整到較大尺寸（較小的圖像用白色背景填充）
        img1_cropped = np.ones((max_h, max_w, 3), dtype=np.uint8) * 255
        img2_cropped = np.ones((max_h, max_w, 3), dtype=np.uint8) * 255
        
        # 將原始圖像複製到左上角
        img1_cropped[0:h1, 0:w1] = img1
        img2_cropped[0:h2, 0:w2] = img2
        
        # 背景移除和透明化處理
        def remove_background_and_make_transparent(img):
            """
            移除背景並創建透明圖像
            使用多種方法檢測背景，優先考慮圖像邊緣的顏色
            
            背景檢測邏輯（像素級處理）：
            1. 檢測圖像邊緣的顏色（至少5像素寬度，對於較大圖像會檢測更多）
            2. 使用中位數計算背景色，對邊緣區域的少量印鑑內容有抗干擾能力
            3. 根據顏色距離和亮度閾值創建背景遮罩
            4. 印章遮罩 = 背景遮罩的反轉（純像素級處理）
            
            可選優化（目前停用）：
            - 形態學操作：可用於填補小洞和去除噪點，但會改變像素級結果
            - 如需啟用，請取消相關註解
            
            注意：此函數需要圖像邊緣有足夠的背景區域（建議至少5像素邊距）。
            多印鑑裁切時使用15像素邊距可以確保邊緣區域主要是背景，提高檢測準確性。
            
            Args:
                img: BGR 圖像（建議邊緣有至少5像素的背景區域）
                
            Returns:
                (mask, rgba_image) - 印章遮罩和帶透明通道的圖像
            """
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 方法1: 檢測圖像邊緣的顏色（通常邊緣是背景）
            # 邊緣檢測寬度：至少5像素，對於較大圖像會檢測更多（最多約圖像尺寸的1/30）
            # 15像素的裁切邊距足夠提供穩定的背景色檢測
            edge_width = max(5, min(h, w) // 30)
            
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
            threshold = max(15, min(40, edge_std * 2))
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
            
            # ===== 形態學操作優化遮罩（可選）=====
            # 說明：形態學操作用於優化遮罩，填補小洞和去除噪點
            # 如果需要純像素級處理，可以跳過此步驟
            # 
            # MORPH_CLOSE（閉運算）：先膨脹後腐蝕，用於填補印章內部的小洞
            #   - 效果：連接斷裂區域，填補背景被誤判為印章的小洞
            #   - kernel: 結構元素大小，越大處理範圍越大
            #   - iterations: 執行次數，越多填補的洞越大
            # 
            # MORPH_OPEN（開運算）：先腐蝕後膨脹，用於去除印章外部的小噪點
            #   - 效果：去除背景被誤判為印章的小噪點，平滑邊緣
            #   - iterations: 執行次數，越多去除的噪點越多（但可能過度侵蝕）
            # 
            # 參數調整建議：
            #   - 如果小洞沒被填補：增加 CLOSE iterations 或使用更大的 kernel (5x5)
            #   - 如果印章被過度侵蝕：減少 OPEN iterations 或完全移除 OPEN
            #   - 如果噪點沒被去除：增加 OPEN iterations
            # 
            # 目前狀態：已停用（註解掉），僅保留像素級處理
            # 如需啟用，取消以下註解：
            # kernel = np.ones((3, 3), np.uint8)
            # seal_mask = cv2.morphologyEx(seal_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            # seal_mask = cv2.morphologyEx(seal_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            # ==========================================
            
            # 確保印章 mask 是二值化的
            _, seal_mask = cv2.threshold(seal_mask, 127, 255, cv2.THRESH_BINARY)
            
            # 創建 RGBA 圖像
            rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = seal_mask
            
            return seal_mask, rgba
        
        # 處理兩個圖像，獲取印章遮罩
        mask1, _ = remove_background_and_make_transparent(img1_cropped)
        mask2, _ = remove_background_and_make_transparent(img2_cropped)
        
        # 轉換為灰度圖以便比對
        gray1 = cv2.cvtColor(img1_cropped, cv2.COLOR_BGR2GRAY) if len(img1_cropped.shape) == 3 else img1_cropped
        gray2 = cv2.cvtColor(img2_cropped, cv2.COLOR_BGR2GRAY) if len(img2_cropped.shape) == 3 else img2_cropped
        
        # 使用 mask 來獲取印章部分的二值化圖像
        binary1 = np.zeros_like(gray1)
        binary1[mask1 > 0] = 255
        
        binary2 = np.zeros_like(gray2)
        binary2[mask2 > 0] = 255
        
        # 計算差異區域
        diff_mask_2_only = (binary2 > 0) & (binary1 == 0)  # 只有圖像2有
        diff_mask_1_only = (binary1 > 0) & (binary2 == 0)  # 只有圖像1有
        overlap_mask = (binary1 > 0) & (binary2 > 0)  # 兩者重疊區域
        
        # 計算像素差異（在重疊區域內）
        # gray_diff: 兩個灰度圖的絕對差值，範圍 0-255 (uint8)
        #   - gray_diff[i,j] = |gray1[i,j] - gray2[i,j]|
        #   - 值越大表示兩個圖像在該像素位置的差異越大
        #   - 0 表示完全相同，255 表示完全相反（黑 vs 白）
        # diff_threshold: 差異閾值，範圍 0-255
        #   - 當 gray_diff > diff_threshold 時，該像素被判定為有差異
        #   - threshold 越小，越敏感，更多像素被標記為差異
        #   - threshold 越大，越嚴格，只有明顯差異的像素被標記
        #   - 預設值 100 約為灰度範圍 (0-255) 的 39%，適合一般情況
        #   對於同一印鑑的比對，使用更寬鬆的閾值（100-120），允許輕微的像素差異
        #   因為同一印鑑蓋出來的圖像可能因為：
        #   1. 蓋印力度不同導致顏色深淺不同
        #   2. 紙張質地不同導致墨水擴散不同
        #   3. 掃描或拍攝條件不同導致亮度對比度不同
        diff_threshold = 100  # 差異閾值 (0-255)，從70提高到100，更適合同一印鑑的比對
        gray_diff = cv2.absdiff(gray1, gray2)  # 灰度差異圖，範圍 0-255
        pixel_diff_mask = (gray_diff > diff_threshold) & overlap_mask  # 重疊區域內的像素差異
        
        # 計算 normalize 範圍（基於 mask1 和 mask2 各自區域內的 gray_diff 最大值和最小值）
        mask1_max = np.max(gray_diff[mask1 > 0]) if np.any(mask1 > 0) else 255
        mask1_min = np.min(gray_diff[mask1 > 0]) if np.any(mask1 > 0) else 0
        mask2_max = np.max(gray_diff[mask2 > 0]) if np.any(mask2 > 0) else 255
        mask2_min = np.min(gray_diff[mask2 > 0]) if np.any(mask2 > 0) else 0
        
        normalize_max = max(mask1_max, mask2_max)
        normalize_min = min(mask1_min, mask2_min)
        
        # Normalize gray_diff 到 0-255 範圍
        if normalize_max > normalize_min:
            gray_diff_normalized = ((gray_diff.astype(np.float32) - normalize_min) / (normalize_max - normalize_min) * 255).astype(np.uint8)
        else:
            # 如果最大值等於最小值，不進行 normalize
            gray_diff_normalized = gray_diff
        
        # 將 normalize 後的 gray_diff 轉換為熱力圖視覺化（使用 COLORMAP_HOT：黑色→紅色→黃色→白色）
        # 值越大（差異越大）顏色越熱（越亮）
        gray_diff_heatmap = cv2.applyColorMap(gray_diff_normalized, cv2.COLORMAP_HOT)
        # 只在重疊區域顯示熱力圖，非重疊區域設為黑色
        overlap_mask_3ch = np.stack([overlap_mask] * 3, axis=2)  # 擴展為3通道
        gray_diff_heatmap = np.where(overlap_mask_3ch, gray_diff_heatmap, 0)

        
        # 計算重疊區域且沒有差異的部分
        overlap_no_diff = overlap_mask & (~pixel_diff_mask)
        
        # 創建彩色疊圖（OpenCV 使用 BGR 格式）
        # 重疊區域用原色呈現，差異區域用黑色標記
        overlay1_on_2 = np.zeros((max_h, max_w, 3), dtype=np.uint8)  # 圖像1疊在圖像2校正上
        overlay2_on_1 = np.zeros((max_h, max_w, 3), dtype=np.uint8)  # 圖像2校正疊在圖像1上
        
        # 差異區域顏色
        diff_color = np.array([0, 0, 0], dtype=np.uint8)  # 黑色（用於獨有區域）
        pixel_diff_color = np.array([0, 165, 255], dtype=np.uint8)  # 亮橘色 (BGR格式)
        
        # 疊圖1：圖像1疊在圖像2校正上
        # 1. 重疊且無差異區域：使用圖像1的原始顏色（圖像1在上層）
        overlay1_on_2[overlap_no_diff] = img1_cropped[overlap_no_diff]
        # 2. 圖像1獨有區域：顯示圖像1的原始顏色
        overlay1_on_2[diff_mask_1_only] = img1_cropped[diff_mask_1_only]
        # 3. 圖像2獨有區域：使用黑色標記（差異區域）
        overlay1_on_2[diff_mask_2_only] = diff_color
        # 4. 重疊區域內的像素差異：使用亮橘色標記
        overlay1_on_2[pixel_diff_mask] = pixel_diff_color
        
        # 疊圖2：圖像2校正疊在圖像1上
        # 1. 重疊且無差異區域：使用圖像2的原始顏色（圖像2在上層）
        overlay2_on_1[overlap_no_diff] = img2_cropped[overlap_no_diff]
        # 2. 圖像2獨有區域：顯示圖像2的原始顏色
        overlay2_on_1[diff_mask_2_only] = img2_cropped[diff_mask_2_only]
        # 3. 圖像1獨有區域：使用黑色標記（差異區域）
        overlay2_on_1[diff_mask_1_only] = diff_color
        # 4. 重疊區域內的像素差異：使用亮橘色標記
        overlay2_on_1[pixel_diff_mask] = pixel_diff_color
        
        # 創建帶透明背景的疊圖（使用 PNG 格式支持透明度）
        overlay1_rgba = np.zeros((max_h, max_w, 4), dtype=np.uint8)
        overlay2_rgba = np.zeros((max_h, max_w, 4), dtype=np.uint8)
        
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
        
        # 驗證文件是否成功保存
        if not overlay1_file.exists() or not overlay2_file.exists():
            print(f"警告：疊圖文件保存失敗 {record_id}: overlay1={overlay1_file.exists()}, overlay2={overlay2_file.exists()}")
            return None, None, None, None, None, None, None
        
        # 保存5個mask圖像和gray_diff視覺化（二值圖像：白色=True區域，黑色=False區域）
        # 將布林mask轉換為uint8格式（True→255白色，False→0黑色）
        overlap_mask_img = overlap_mask.astype(np.uint8) * 255
        pixel_diff_mask_img = pixel_diff_mask.astype(np.uint8) * 255
        diff_mask_2_only_img = diff_mask_2_only.astype(np.uint8) * 255
        diff_mask_1_only_img = diff_mask_1_only.astype(np.uint8) * 255
        
        # 保存mask圖像和gray_diff視覺化
        overlap_mask_file = overlay_dir / f"overlap_mask_{record_id}.png"
        pixel_diff_mask_file = overlay_dir / f"pixel_diff_mask_{record_id}.png"
        diff_mask_2_only_file = overlay_dir / f"diff_mask_2_only_{record_id}.png"
        diff_mask_1_only_file = overlay_dir / f"diff_mask_1_only_{record_id}.png"
        gray_diff_file = overlay_dir / f"gray_diff_{record_id}.png"
        
        try:
            cv2.imwrite(str(overlap_mask_file), overlap_mask_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            cv2.imwrite(str(pixel_diff_mask_file), pixel_diff_mask_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            cv2.imwrite(str(diff_mask_2_only_file), diff_mask_2_only_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            cv2.imwrite(str(diff_mask_1_only_file), diff_mask_1_only_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            
            # 直接保存 gray_diff 熱力圖（不包含 legend）
            cv2.imwrite(str(gray_diff_file), gray_diff_heatmap, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            
            # 驗證mask文件和gray_diff文件是否成功保存
            mask_files_exist = all([
                overlap_mask_file.exists(),
                pixel_diff_mask_file.exists(),
                diff_mask_2_only_file.exists(),
                diff_mask_1_only_file.exists(),
                gray_diff_file.exists()
            ])
            
            if not mask_files_exist:
                print(f"警告：mask或gray_diff文件保存失敗 {record_id}")
                return str(overlay1_file), str(overlay2_file), None, None, None, None, None
            
            # 返回絕對路徑（用於資料庫存儲）
            return (
                str(overlay1_file),
                str(overlay2_file),
                str(overlap_mask_file),
                str(pixel_diff_mask_file),
                str(diff_mask_2_only_file),
                str(diff_mask_1_only_file),
                str(gray_diff_file)
            )
        except Exception as e:
            print(f"警告：保存mask或gray_diff圖像失敗 {record_id}: {e}")
            # 即使mask保存失敗，也返回overlay圖像
            return str(overlay1_file), str(overlay2_file), None, None, None, None, None
        
    except Exception as e:
        print(f"錯誤：無法生成疊圖 {record_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None


def calculate_mask_statistics(
    overlap_mask_path: Optional[str],
    pixel_diff_mask_path: Optional[str],
    diff_mask_2_only_path: Optional[str],
    diff_mask_1_only_path: Optional[str]
) -> Dict[str, Any]:
    """
    計算4個mask圖像的像素統計資訊
    
    Args:
        overlap_mask_path: 重疊區域mask路徑
        pixel_diff_mask_path: 像素差異mask路徑
        diff_mask_2_only_path: 圖像2獨有區域mask路徑
        diff_mask_1_only_path: 圖像1獨有區域mask路徑
        
    Returns:
        包含像素統計資訊的字典：
        {
            'overlap_pixels': int,
            'pixel_diff_pixels': int,
            'diff_2_only_pixels': int,
            'diff_1_only_pixels': int,
            'total_seal_pixels': int,
            'overlap_ratio': float,
            'pixel_diff_ratio': float,
            'diff_2_only_ratio': float,
            'diff_1_only_ratio': float,
            'total_diff_ratio': float
        }
    """
    try:
        # 轉換路徑（處理容器路徑）
        def normalize_path(p):
            if not p:
                return None
            s = str(p)
            if s.startswith('/app/'):
                s = s.replace('/app/', '')
            return Path(s)
        
        overlap_path = normalize_path(overlap_mask_path) if overlap_mask_path else None
        pixel_diff_path = normalize_path(pixel_diff_mask_path) if pixel_diff_mask_path else None
        diff_2_only_path = normalize_path(diff_mask_2_only_path) if diff_mask_2_only_path else None
        diff_1_only_path = normalize_path(diff_mask_1_only_path) if diff_mask_1_only_path else None
        
        # 讀取mask圖像（二值圖像：白色=255，黑色=0）
        overlap_mask = None
        pixel_diff_mask = None
        diff_mask_2_only = None
        diff_mask_1_only = None
        
        if overlap_path and overlap_path.exists():
            overlap_mask = cv2.imread(str(overlap_path), cv2.IMREAD_GRAYSCALE)
            if overlap_mask is not None:
                overlap_mask = overlap_mask > 127  # 轉換為布林mask
        
        if pixel_diff_path and pixel_diff_path.exists():
            pixel_diff_mask = cv2.imread(str(pixel_diff_path), cv2.IMREAD_GRAYSCALE)
            if pixel_diff_mask is not None:
                pixel_diff_mask = pixel_diff_mask > 127
        
        if diff_2_only_path and diff_2_only_path.exists():
            diff_mask_2_only = cv2.imread(str(diff_2_only_path), cv2.IMREAD_GRAYSCALE)
            if diff_mask_2_only is not None:
                diff_mask_2_only = diff_mask_2_only > 127
        
        if diff_1_only_path and diff_1_only_path.exists():
            diff_mask_1_only = cv2.imread(str(diff_1_only_path), cv2.IMREAD_GRAYSCALE)
            if diff_mask_1_only is not None:
                diff_mask_1_only = diff_mask_1_only > 127
        
        # 計算像素數量
        overlap_pixels = int(np.sum(overlap_mask)) if overlap_mask is not None else 0
        pixel_diff_pixels = int(np.sum(pixel_diff_mask)) if pixel_diff_mask is not None else 0
        diff_2_only_pixels = int(np.sum(diff_mask_2_only)) if diff_mask_2_only is not None else 0
        diff_1_only_pixels = int(np.sum(diff_mask_1_only)) if diff_mask_1_only is not None else 0
        
        # 計算總印章像素數量（所有mask的聯集）
        # 創建一個合併的mask來計算總像素數
        total_seal_pixels = 0
        if overlap_mask is not None or pixel_diff_mask is not None or diff_mask_2_only is not None or diff_mask_1_only is not None:
            # 獲取圖像尺寸（從第一個可用的mask）
            h, w = 0, 0
            for mask in [overlap_mask, pixel_diff_mask, diff_mask_2_only, diff_mask_1_only]:
                if mask is not None:
                    h, w = mask.shape[:2]
                    break
            
            if h > 0 and w > 0:
                # 創建合併mask
                combined_mask = np.zeros((h, w), dtype=bool)
                if overlap_mask is not None:
                    combined_mask = combined_mask | overlap_mask
                if pixel_diff_mask is not None:
                    combined_mask = combined_mask | pixel_diff_mask
                if diff_mask_2_only is not None:
                    combined_mask = combined_mask | diff_mask_2_only
                if diff_mask_1_only is not None:
                    combined_mask = combined_mask | diff_mask_1_only
                
                total_seal_pixels = int(np.sum(combined_mask))
        
        # 計算比例
        overlap_ratio = overlap_pixels / total_seal_pixels if total_seal_pixels > 0 else 0.0
        pixel_diff_ratio = pixel_diff_pixels / overlap_pixels if overlap_pixels > 0 else 1.0  # 如果沒有重疊，設為最大懲罰
        diff_2_only_ratio = diff_2_only_pixels / total_seal_pixels if total_seal_pixels > 0 else 0.0
        diff_1_only_ratio = diff_1_only_pixels / total_seal_pixels if total_seal_pixels > 0 else 0.0
        total_diff_ratio = (pixel_diff_pixels + diff_2_only_pixels + diff_1_only_pixels) / total_seal_pixels if total_seal_pixels > 0 else 0.0
        
        return {
            'overlap_pixels': overlap_pixels,
            'pixel_diff_pixels': pixel_diff_pixels,
            'diff_2_only_pixels': diff_2_only_pixels,
            'diff_1_only_pixels': diff_1_only_pixels,
            'total_seal_pixels': total_seal_pixels,
            'overlap_ratio': float(overlap_ratio),
            'pixel_diff_ratio': float(pixel_diff_ratio),
            'diff_2_only_ratio': float(diff_2_only_ratio),
            'diff_1_only_ratio': float(diff_1_only_ratio),
            'total_diff_ratio': float(total_diff_ratio)
        }
        
    except Exception as e:
        print(f"錯誤：計算mask統計資訊失敗: {e}")
        import traceback
        traceback.print_exc()
        # 返回默認值
        return {
            'overlap_pixels': 0,
            'pixel_diff_pixels': 0,
            'diff_2_only_pixels': 0,
            'diff_1_only_pixels': 0,
            'total_seal_pixels': 0,
            'overlap_ratio': 0.0,
            'pixel_diff_ratio': 1.0,
            'diff_2_only_ratio': 0.0,
            'diff_1_only_ratio': 0.0,
            'total_diff_ratio': 0.0
        }


def calculate_mask_based_similarity(
    mask_stats: Dict[str, Any],
    overlap_weight: float = 0.5,
    pixel_diff_penalty_weight: float = 0.3,
    unique_region_penalty_weight: float = 0.2
) -> float:
    """
    基於mask區域的加權相似度計算
    
    演算法設計：
    1. 重疊區域獎勵：重疊區域比例越高，相似度越高
    2. 像素差異懲罰：重疊區域內的像素差異比例越高，相似度越低
    3. 獨有區域懲罰：獨有區域比例越高，相似度越低
    
    Args:
        mask_stats: 由 calculate_mask_statistics 返回的統計資訊字典
        overlap_weight: 重疊區域權重，預設為 0.5
        pixel_diff_penalty_weight: 像素差異懲罰權重，預設為 0.3
        unique_region_penalty_weight: 獨有區域懲罰權重，預設為 0.2
        
    Returns:
        相似度分數 (0.0-1.0)
    """
    try:
        # 邊界處理：如果總印章像素為0，返回0.0
        if mask_stats.get('total_seal_pixels', 0) == 0:
            return 0.0
        
        # 1. 重疊區域獎勵
        overlap_ratio = mask_stats.get('overlap_ratio', 0.0)
        overlap_score = overlap_ratio
        
        # 2. 像素差異懲罰（改進：使用更寬鬆的懲罰機制）
        pixel_diff_ratio = mask_stats.get('pixel_diff_ratio', 1.0)
        # 對於同一印鑑，即使有像素差異，也應該給予較高的分數
        # 使用平方根函數減緩懲罰強度：當 pixel_diff_ratio 較小時，懲罰較輕
        pixel_diff_penalty = 1.0 - (pixel_diff_ratio ** 0.7)  # 使用0.7次方減緩懲罰
        
        # 3. 獨有區域懲罰（改進：對小範圍獨有區域更寬容）
        diff_1_only_ratio = mask_stats.get('diff_1_only_ratio', 0.0)
        diff_2_only_ratio = mask_stats.get('diff_2_only_ratio', 0.0)
        total_unique_ratio = diff_1_only_ratio + diff_2_only_ratio
        # 對於同一印鑑，對齊不完美時會產生少量獨有區域，使用平方根函數減緩懲罰
        unique_penalty = 1.0 - (total_unique_ratio ** 0.6)  # 使用0.6次方減緩懲罰
        
        # 最終相似度計算（改進：提高重疊區域權重，降低懲罰權重）
        # 對於同一印鑑，重疊區域應該是最重要的指標
        similarity = (
            overlap_score * overlap_weight +
            pixel_diff_penalty * pixel_diff_penalty_weight +
            unique_penalty * unique_region_penalty_weight
        )
        
        # 額外優化：如果重疊區域比例很高（>80%），即使有像素差異和獨有區域，也應該給予較高分數
        # 這適用於同一印鑑但對齊略有偏差的情況
        if overlap_ratio > 0.8:
            # 使用重疊區域作為主要指標，其他指標作為微調
            similarity = overlap_ratio * 0.7 + similarity * 0.3
        
        # 確保返回值在 [0.0, 1.0] 範圍內
        similarity = max(0.0, min(1.0, similarity))
        
        return float(similarity)
        
    except Exception as e:
        print(f"錯誤：計算mask-based相似度失敗: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

