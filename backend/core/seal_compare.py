"""
印鑑比對模組
用於比對兩個印章圖像是否完全一致
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import hashlib


class SealComparator:
    """印鑑比對器"""
    
    def __init__(self, threshold: float = 0.95):
        """
        初始化比對器
        
        Args:
            threshold: 相似度閾值，預設為 0.95（95%）
        """
        self.threshold = threshold
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        載入圖像
        
        Args:
            image_path: 圖像路徑
            
        Returns:
            圖像陣列，如果載入失敗則返回 None
        """
        if not Path(image_path).exists():
            print(f"錯誤：找不到圖像文件 {image_path}")
            return None
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"錯誤：無法讀取圖像文件 {image_path}")
            return None
        
        return image
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        預處理圖像（轉換為灰度圖、亮度/對比度標準化、二值化）
        注意：不進行尺寸調整，保持原始掃描尺寸
        
        Args:
            image: 原始圖像
            
        Returns:
            預處理後的圖像
        """
        if image is None:
            raise ValueError("圖像不能為 None")
        
        if image.size == 0:
            raise ValueError("圖像不能為空")
        
        # 轉換為灰度圖
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray = image
        else:
            raise ValueError(f"不支持的圖像維度: {image.shape}")
        
        # 不進行尺寸調整，保持原始掃描尺寸（假設使用同一台掃描機，尺寸一致）
        # 亮度/對比度標準化：使用 CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # 這可以減少不同掃描品質造成的差異
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = clahe.apply(gray)
        
        # 二值化處理：嘗試多種方法，選擇最適合的
        # 方法1: OTSU
        _, binary_otsu = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 方法2: 自適應閾值（對不同區域使用不同閾值）
        binary_adaptive = cv2.adaptiveThreshold(
            normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 選擇邊緣更清晰的二值化結果
        # 計算兩種方法的邊緣數量，選擇邊緣數量適中的
        edges_otsu = cv2.Canny(binary_otsu, 50, 150)
        edges_adaptive = cv2.Canny(binary_adaptive, 50, 150)
        
        edge_count_otsu = np.count_nonzero(edges_otsu)
        edge_count_adaptive = np.count_nonzero(edges_adaptive)
        
        # 選擇邊緣數量更合理的（避免過多或過少）
        target_edge_ratio = 0.05
        total_pixels = binary_otsu.size
        
        otsu_ratio = edge_count_otsu / total_pixels
        adaptive_ratio = edge_count_adaptive / total_pixels
        
        if abs(otsu_ratio - target_edge_ratio) < abs(adaptive_ratio - target_edge_ratio):
            binary = binary_otsu
        else:
            binary = binary_adaptive
        
        return binary
    
    def _auto_detect_bounds_and_remove_background(self, image: np.ndarray) -> np.ndarray:
        """
        自動偵測圖像外框並移除背景顏色
        
        使用演算法自動偵測圖像的實際內容邊界，並移除背景（通常是白紙的顏色）
        
        Args:
            image: 輸入圖像（可以是彩色或灰度）
            
        Returns:
            裁切並移除背景後的圖像（背景設為白色 255）
        """
        if image is None or image.size == 0:
            return image
        
        # 轉換為灰度圖（如果需要的話）
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        h, w = gray.shape
        
        # 方法1: 檢測背景顏色（通常是白色，值接近255）
        # 分析圖像邊緣的顏色（邊緣通常是背景）
        edge_width = max(5, min(h, w) // 20)
        edge_pixels = []
        edge_pixels.extend(gray[0:edge_width, :].flatten())
        edge_pixels.extend(gray[h-edge_width:h, :].flatten())
        edge_pixels.extend(gray[:, 0:edge_width].flatten())
        edge_pixels.extend(gray[:, w-edge_width:w].flatten())
        
        # 計算背景顏色的中位數和標準差
        edge_pixels_array = np.array(edge_pixels)
        bg_color = np.median(edge_pixels_array)
        bg_std = np.std(edge_pixels_array)
        
        # 背景閾值：背景顏色 ± 2倍標準差（通常背景是白色，值接近255）
        bg_threshold_low = max(200, int(bg_color - 2 * bg_std))
        bg_threshold_high = 255
        
        # 方法2: 使用 OTSU 二值化來分離前景和背景
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 方法3: 結合背景顏色檢測和二值化結果
        # 創建背景遮罩：接近背景顏色的像素
        bg_mask_color = (gray >= bg_threshold_low) & (gray <= bg_threshold_high)
        
        # 結合 OTSU 結果：OTSU 的前景區域（值為255）應該保留
        # OTSU 的前景是255，背景是0，所以背景遮罩應該是 OTSU 背景（0）且顏色接近背景
        bg_mask_otsu = (binary_otsu == 0)  # OTSU 的背景區域
        
        # 結合兩個條件：既是背景顏色，又是 OTSU 識別的背景
        bg_mask = bg_mask_color & bg_mask_otsu
        
        # 形態學操作：去除小的噪點
        kernel = np.ones((3, 3), np.uint8)
        bg_mask = cv2.morphologyEx(bg_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=2)
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        bg_mask = bg_mask > 0
        
        # 方法4: 使用輪廓檢測找到實際內容的邊界框
        # 創建前景遮罩（非背景區域）
        foreground_mask = ~bg_mask
        
        # 形態學操作優化前景遮罩
        foreground_mask = cv2.morphologyEx(foreground_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=3)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 找到輪廓
        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            # 如果沒有找到輪廓，返回原圖
            return image
        
        # 找到最大的輪廓（假設是印章）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 檢查輪廓面積是否足夠大（至少佔圖像的1%）
        contour_area = cv2.contourArea(largest_contour)
        min_area = (h * w) * 0.01
        if contour_area < min_area:
            # 輪廓太小，可能是噪點，返回原圖
            return image
        
        # 計算邊界框
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 確保邊界框有效
        if w <= 0 or h <= 0:
            return image
        
        # 添加一些邊距（5%）
        margin = max(5, min(h, w) // 20)
        x = max(0, x - margin)
        y = max(0, y - margin)
        # 確保不會超出圖像邊界
        w = min(w + 2 * margin, gray.shape[1] - x)
        h = min(h + 2 * margin, gray.shape[0] - y)
        
        # 再次確保邊界框有效
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
            return image
        
        # 裁切圖像到邊界框
        if len(image.shape) == 3:
            cropped = image[y:y+h, x:x+w].copy()
        else:
            cropped = gray[y:y+h, x:x+w].copy()
        
        # 移除背景：將背景區域設為白色（255）
        if len(cropped.shape) == 3:
            # 彩色圖像：轉換為灰度來檢測背景
            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        else:
            cropped_gray = cropped.copy()
        
        # 在裁切後的圖像上重新檢測背景並移除
        if cropped.size == 0:
            return image
        
        edge_width_crop = max(3, min(h, w) // 30)
        edge_pixels_crop = []
        
        # 確保圖像足夠大才能進行邊緣檢測
        if h > edge_width_crop * 2 and w > edge_width_crop * 2:
            try:
                edge_pixels_crop.extend(cropped_gray[0:edge_width_crop, :].flatten())
                edge_pixels_crop.extend(cropped_gray[h-edge_width_crop:h, :].flatten())
                edge_pixels_crop.extend(cropped_gray[:, 0:edge_width_crop].flatten())
                edge_pixels_crop.extend(cropped_gray[:, w-edge_width_crop:w].flatten())
                
                if len(edge_pixels_crop) > 0:
                    bg_color_crop = np.median(np.array(edge_pixels_crop))
                    bg_threshold_crop = max(200, int(bg_color_crop - 30))
                    
                    # 創建背景遮罩
                    bg_mask_crop = (cropped_gray >= bg_threshold_crop)
                    
                    # 將背景設為白色
                    result = cropped.copy()
                    if len(cropped.shape) == 3:
                        result[bg_mask_crop] = [255, 255, 255]
                    else:
                        result[bg_mask_crop] = 255
                else:
                    result = cropped
            except Exception:
                # 如果處理出錯，返回裁切後的圖像
                result = cropped
        else:
            # 圖像太小，直接返回裁切結果
            result = cropped
        
        return result
    
    def find_best_rotation_angle(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        找到圖像2相對於圖像1的最佳旋轉角度（優化算法）
        
        使用粗到細（coarse-to-fine）搜索策略，搜索範圍限制在-80到+80度：
        1. 第一階段：每10度搜索一次，找到最佳角度範圍（-80到+80度，共17次）
        2. 第二階段：在最佳角度±10度範圍內，每2度搜索一次
        3. 第三階段：在最佳角度±2度範圍內，每0.5度搜索一次
        
        使用重疊相似度作為評估標準，找到與圖像1重疊最大的角度。
        評估方法：結合 SSIM、模板匹配和重疊區域計算，最大化重疊相似度。
        
        Args:
            img1: 參考圖像（已預處理）
            img2: 待旋轉圖像（已預處理）
            
        Returns:
            (最佳角度, 旋轉後的圖像2)
        """
        # 使用縮小圖像進行快速評估（加速搜索）
        scale_factor = 0.3  # 縮小到30%以加快速度
        h, w = img1.shape
        small_h, small_w = int(h * scale_factor), int(w * scale_factor)
        
        # 確保最小尺寸
        if small_h < 50 or small_w < 50:
            scale_factor = max(50 / h, 50 / w)
            small_h, small_w = int(h * scale_factor), int(w * scale_factor)
        
        img1_small = cv2.resize(img1, (small_w, small_h), interpolation=cv2.INTER_AREA)
        img2_small = cv2.resize(img2, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        # 確保尺寸相同
        if img1_small.shape != img2_small.shape:
            target_h = max(img1_small.shape[0], img2_small.shape[0])
            target_w = max(img1_small.shape[1], img2_small.shape[1])
            img1_small = cv2.resize(img1_small, (target_w, target_h))
            img2_small = cv2.resize(img2_small, (target_w, target_h))
        
        center = (small_w // 2, small_h // 2)
        
        # 計算重疊相似度的輔助函數
        def calculate_overlap_similarity(img1_ref, img2_rot):
            """計算兩個圖像的重疊相似度（結合多種指標）"""
            # 確保尺寸相同
            if img1_ref.shape != img2_rot.shape:
                target_h = max(img1_ref.shape[0], img2_rot.shape[0])
                target_w = max(img1_ref.shape[1], img2_rot.shape[1])
                img1_ref = cv2.resize(img1_ref, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                img2_rot = cv2.resize(img2_rot, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            # 1. SSIM 相似度（結構相似性）
            ssim = self._calculate_ssim(img1_ref, img2_rot)
            
            # 2. 模板匹配相似度
            template_match = self._template_match(img1_ref, img2_rot)
            
            # 3. 重疊區域計算（非背景像素的重疊率）
            # 將背景（255）視為無效區域
            mask1 = (img1_ref < 250).astype(np.uint8)  # 非背景區域
            mask2 = (img2_rot < 250).astype(np.uint8)  # 非背景區域
            overlap_mask = mask1 & mask2  # 重疊區域
            union_mask = mask1 | mask2  # 並集區域
            
            if np.sum(union_mask) > 0:
                overlap_ratio = np.sum(overlap_mask) / np.sum(union_mask)
            else:
                overlap_ratio = 0.0
            
            # 4. 在重疊區域內的像素相似度
            if np.sum(overlap_mask) > 0:
                overlap_pixels1 = img1_ref[overlap_mask > 0]
                overlap_pixels2 = img2_rot[overlap_mask > 0]
                pixel_sim = 1.0 - np.mean(np.abs(overlap_pixels1.astype(np.float32) - overlap_pixels2.astype(np.float32))) / 255.0
                pixel_sim = max(0.0, min(1.0, pixel_sim))
            else:
                pixel_sim = 0.0
            
            # 綜合相似度：重疊率權重較高，因為目標是最大化重疊
            similarity = (
                overlap_ratio * 0.40 +      # 重疊區域比例（最重要）
                ssim * 0.25 +                # 結構相似性
                template_match * 0.20 +      # 模板匹配
                pixel_sim * 0.15             # 重疊區域像素相似度
            )
            
            return float(similarity)
        
        # 第一階段：粗搜索（-80到+80度，每10度）
        best_angle_coarse = 0.0
        best_similarity_coarse = 0.0  # 使用重疊相似度，越大越好
        
        for angle in range(-80, 81, 10):
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img2_rotated = cv2.warpAffine(img2_small, M, (small_w, small_h),
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=255)
            similarity = calculate_overlap_similarity(img1_small, img2_rotated)
            
            if similarity > best_similarity_coarse:
                best_similarity_coarse = similarity
                best_angle_coarse = angle
            
            # 早期退出：如果相似度已經很高
            if best_similarity_coarse > 0.99:
                break
        
        # 第二階段：中等搜索（在最佳角度±10度範圍內，每2度）
        best_angle_medium = best_angle_coarse
        best_similarity_medium = best_similarity_coarse
        
        search_range = range(int(best_angle_coarse - 10), int(best_angle_coarse + 11), 2)
        
        for angle in search_range:
            # 限制在-80到+80度範圍內
            if angle < -80 or angle > 80:
                continue
                
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img2_rotated = cv2.warpAffine(img2_small, M, (small_w, small_h),
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=255)
            similarity = calculate_overlap_similarity(img1_small, img2_rotated)
            
            if similarity > best_similarity_medium:
                best_similarity_medium = similarity
                best_angle_medium = angle
        
        # 第三階段：細搜索（在最佳角度±2度範圍內，每0.5度）
        best_angle = best_angle_medium
        best_similarity = best_similarity_medium
        
        offsets = np.arange(-2.0, 2.5, 0.5)
        for offset in offsets:
            angle = best_angle_medium + offset
            
            # 限制在-80到+80度範圍內
            if angle < -80 or angle > 80:
                continue
                
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img2_rotated = cv2.warpAffine(img2_small, M, (small_w, small_h),
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=255)
            similarity = calculate_overlap_similarity(img1_small, img2_rotated)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_angle = angle
            
            # 早期退出：如果相似度已經很高
            if best_similarity > 0.99:
                break
        
        # 第四階段：超細搜索（在最佳角度±1度範圍內，每0.1度）
        if best_similarity < 0.99:  # 如果相似度還不夠高，進行超細搜索
            best_angle_ultra = best_angle
            best_similarity_ultra = best_similarity
            
            offsets_ultra = np.arange(-1.0, 1.1, 0.1)
            for offset in offsets_ultra:
                angle = best_angle + offset
                
                # 限制在-80到+80度範圍內
                if angle < -80 or angle > 80:
                    continue
                    
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img2_rotated = cv2.warpAffine(img2_small, M, (small_w, small_h),
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=255)
                similarity = calculate_overlap_similarity(img1_small, img2_rotated)
                
                if similarity > best_similarity_ultra:
                    best_similarity_ultra = similarity
                    best_angle_ultra = angle
                
                if best_similarity_ultra > 0.99:
                    break
            
            best_angle = best_angle_ultra
            best_similarity = best_similarity_ultra
        
        # 使用最佳角度旋轉原始尺寸的圖像2
        h2, w2 = img2.shape
        center_full = (w2 // 2, h2 // 2)
        M = cv2.getRotationMatrix2D(center_full, best_angle, 1.0)
        img2_rotated_full = cv2.warpAffine(img2, M, (w2, h2),
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=255)
        
        return best_angle, img2_rotated_full
    
    def find_best_translation(self, img1: np.ndarray, img2: np.ndarray, 
                              max_offset: int = 200) -> Tuple[Tuple[int, int], np.ndarray]:
        """
        找到圖像2相對於圖像1的最佳平移偏移量
        
        使用多尺度搜索策略（粗到細）：
        1. 粗搜索：在縮小圖像上使用模板匹配快速找到大致位置
        2. 細搜索：在最佳位置附近精確搜索
        
        Args:
            img1: 參考圖像（已預處理）
            img2: 待平移圖像（已預處理）
            max_offset: 最大平移偏移量（像素），默認200
            
        Returns:
            ((最佳x偏移, 最佳y偏移), 平移後的圖像2)
        """
        h, w = img1.shape
        
        # 使用縮小圖像進行快速評估
        scale_factor = 0.2  # 縮小到20%以加快速度
        small_h, small_w = int(h * scale_factor), int(w * scale_factor)
        
        # 確保最小尺寸
        if small_h < 50 or small_w < 50:
            scale_factor = max(50 / h, 50 / w)
            small_h, small_w = int(h * scale_factor), int(w * scale_factor)
        
        img1_small = cv2.resize(img1, (small_w, small_h), interpolation=cv2.INTER_AREA)
        img2_small = cv2.resize(img2, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        # 確保尺寸相同
        if img1_small.shape != img2_small.shape:
            target_h = max(img1_small.shape[0], img2_small.shape[0])
            target_w = max(img1_small.shape[1], img2_small.shape[1])
            img1_small = cv2.resize(img1_small, (target_w, target_h))
            img2_small = cv2.resize(img2_small, (target_w, target_h))
        
        # 計算縮小圖像上的最大偏移
        small_max_offset = int(max_offset * scale_factor)
        
        # 第一階段：粗搜索（使用模板匹配）
        # 擴展圖像以允許平移
        expanded_size = (small_h + 2 * small_max_offset, small_w + 2 * small_max_offset)
        img2_expanded = np.ones(expanded_size, dtype=np.uint8) * 255
        start_y = small_max_offset
        start_x = small_max_offset
        img2_expanded[start_y:start_y+small_h, start_x:start_x+small_w] = img2_small
        
        # 模板匹配
        result = cv2.matchTemplate(img2_expanded, img1_small, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # 計算粗搜索的最佳偏移（相對於圖像中心）
        best_offset_x_small = max_loc[0] - small_max_offset
        best_offset_y_small = max_loc[1] - small_max_offset
        
        # 轉換回原始尺寸的偏移
        best_offset_x_coarse = int(best_offset_x_small / scale_factor)
        best_offset_y_coarse = int(best_offset_y_small / scale_factor)
        
        # 限制在最大偏移範圍內
        best_offset_x_coarse = max(-max_offset, min(max_offset, best_offset_x_coarse))
        best_offset_y_coarse = max(-max_offset, min(max_offset, best_offset_y_coarse))
        
        # 第二階段：細搜索（在最佳位置附近精確搜索）
        best_offset_x = best_offset_x_coarse
        best_offset_y = best_offset_y_coarse
        best_score = max_val
        
        # 第二階段：細搜索（在最佳位置附近精確搜索）
        # 擴大搜索範圍並使用更細的步長
        fine_range = 30  # 從20擴大到30
        fine_step = 2    # 從5縮小到2，提高精度
        
        for dy in range(-fine_range, fine_range + 1, fine_step):
            for dx in range(-fine_range, fine_range + 1, fine_step):
                offset_x = best_offset_x_coarse + dx
                offset_y = best_offset_y_coarse + dy
                
                # 限制範圍
                if abs(offset_x) > max_offset or abs(offset_y) > max_offset:
                    continue
                
                # 應用平移並評估
                M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                img2_translated = cv2.warpAffine(img2, M, (w, h),
                                                borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=255)
                
                # 快速評估（使用縮小的圖像）
                img2_translated_small = cv2.resize(img2_translated, (small_w, small_h), 
                                                   interpolation=cv2.INTER_AREA)
                score = self._fast_translation_match(img1_small, img2_translated_small)
                
                if score > best_score:
                    best_score = score
                    best_offset_x = offset_x
                    best_offset_y = offset_y
                
                # 早期退出
                if best_score > 0.98:
                    break
            
            if best_score > 0.98:
                break
        
        # 第三階段：超細搜索（如果分數還不夠高）
        if best_score < 0.95:
            ultra_fine_range = 10
            ultra_fine_step = 1
            
            for dy in range(-ultra_fine_range, ultra_fine_range + 1, ultra_fine_step):
                for dx in range(-ultra_fine_range, ultra_fine_range + 1, ultra_fine_step):
                    offset_x = best_offset_x + dx
                    offset_y = best_offset_y + dy
                    
                    # 限制範圍
                    if abs(offset_x) > max_offset or abs(offset_y) > max_offset:
                        continue
                    
                    # 應用平移並評估
                    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                    img2_translated = cv2.warpAffine(img2, M, (w, h),
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=255)
                    
                    # 快速評估
                    img2_translated_small = cv2.resize(img2_translated, (small_w, small_h), 
                                                       interpolation=cv2.INTER_AREA)
                    score = self._fast_translation_match(img1_small, img2_translated_small)
                    
                    if score > best_score:
                        best_score = score
                        best_offset_x = offset_x
                        best_offset_y = offset_y
                    
                    if best_score > 0.98:
                        break
                
                if best_score > 0.98:
                    break
        
        # 應用最佳平移到原始圖像
        M = np.float32([[1, 0, best_offset_x], [0, 1, best_offset_y]])
        img2_translated_full = cv2.warpAffine(img2, M, (w, h),
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=255)
        
        return (best_offset_x, best_offset_y), img2_translated_full
    
    def _fast_translation_match(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        快速平移匹配評估（改進版）
        
        使用多種方法組合：
        1. 模板匹配（對平移最敏感）
        2. SSIM相似度（結構相似性）
        3. 像素相似度
        
        Args:
            img1: 參考圖像
            img2: 待比對圖像
            
        Returns:
            匹配分數 (0-1)
        """
        # 確保尺寸相同
        if img1.shape != img2.shape:
            target_h = max(img1.shape[0], img2.shape[0])
            target_w = max(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # 方法1：模板匹配（對平移最敏感）
        result = cv2.matchTemplate(img2, img1, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        template_match = float(max_val)
        
        # 方法2：簡化版SSIM
        img1_float = img1.astype(np.float64)
        img2_float = img2.astype(np.float64)
        
        mu1 = cv2.GaussianBlur(img1_float, (5, 5), 1.0)
        mu2 = cv2.GaussianBlur(img2_float, (5, 5), 1.0)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1_float ** 2, (5, 5), 1.0) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2_float ** 2, (5, 5), 1.0) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1_float * img2_float, (5, 5), 1.0) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_score = float(np.mean(ssim_map))
        
        # 方法3：像素相似度
        mask1 = img1 < 250
        mask2 = img2 < 250
        mask_combined = mask1 & mask2
        
        pixel_similarity = 0.0
        if np.sum(mask_combined) > 0:
            diff = cv2.absdiff(img1, img2)
            pixel_similarity = 1.0 - (np.sum(diff[mask_combined] > 10) / np.sum(mask_combined))
        
        # 加權組合
        score = template_match * 0.5 + ssim_score * 0.35 + pixel_similarity * 0.15
        
        return max(0.0, min(1.0, score))  # 確保分數在0-1範圍內
    
    def find_best_rotation_and_translation(
        self, 
        img1: np.ndarray, 
        img2: np.ndarray,
        max_translation: int = 200
    ) -> Tuple[float, Tuple[int, int], np.ndarray]:
        """
        聯合優化旋轉和平移（混合策略）
        
        策略：
        1. 粗搜索階段：順序搜索（先旋轉後平移）
        2. 細搜索階段：聯合優化（同時調整旋轉和平移）
        
        Args:
            img1: 參考圖像（已預處理）
            img2: 待校正圖像（已預處理）
            max_translation: 最大平移偏移量（像素）
            
        Returns:
            (最佳角度, (最佳x偏移, 最佳y偏移), 校正後的圖像2)
        """
        # 階段1：粗搜索（順序）
        # 1.1 先找到最佳旋轉角度
        best_angle, img2_rotated = self.find_best_rotation_angle(img1, img2)
        
        # 1.2 在最佳旋轉基礎上找到最佳平移
        (best_offset_x, best_offset_y), img2_rotated_translated = self.find_best_translation(
            img1, img2_rotated, max_translation
        )
        
        # 階段2：細搜索（聯合優化）- 改進版
        # 擴大搜索範圍：從±3度擴大到±5度，從±20像素擴大到±30像素
        # 使用更細的步長：角度0.5度，平移2像素
        best_angle_final = best_angle
        best_offset_x_final = best_offset_x
        best_offset_y_final = best_offset_y
        best_score = 0.0
        
        # 使用縮小圖像進行快速評估
        scale_factor = 0.3
        h, w = img1.shape
        small_h, small_w = int(h * scale_factor), int(w * scale_factor)
        img1_small = cv2.resize(img1, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        # 細搜索範圍（擴大並使用更細步長）
        angle_range = np.arange(-5.0, 5.5, 0.5)  # 每0.5度（從±3度擴大到±5度）
        offset_range = range(-30, 31, 2)  # 每2像素（從±20擴大到±30，步長從5縮小到2）
        
        for angle_offset in angle_range:
            angle = best_angle + angle_offset
            # 限制在-80到+80度範圍內
            if angle < -80 or angle > 80:
                continue
            
            for dx in offset_range:
                for dy in offset_range:
                    offset_x = best_offset_x + dx
                    offset_y = best_offset_y + dy
                    
                    # 限制範圍
                    if abs(offset_x) > max_translation or abs(offset_y) > max_translation:
                        continue
                    
                    # 應用旋轉和平移
                    h2, w2 = img2.shape
                    center = (w2 // 2, h2 // 2)
                    
                    # 先旋轉
                    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                    img2_rot = cv2.warpAffine(img2, M_rot, (w2, h2),
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=255)
                    
                    # 再平移
                    M_trans = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                    img2_final = cv2.warpAffine(img2_rot, M_trans, (w2, h2),
                                               borderMode=cv2.BORDER_CONSTANT,
                                               borderValue=255)
                    
                    # 快速評估
                    img2_small = cv2.resize(img2_final, (small_w, small_h), 
                                           interpolation=cv2.INTER_AREA)
                    score = self._fast_rotation_match(img1_small, img2_small)
                    
                    if score > best_score:
                        best_score = score
                        best_angle_final = angle
                        best_offset_x_final = offset_x
                        best_offset_y_final = offset_y
                    
                    # 早期退出
                    if best_score > 0.98:
                        break
                
                if best_score > 0.98:
                    break
            
            if best_score > 0.98:
                break
        
        # 階段3：迭代優化（如果找到改善，在最佳值附近再次細搜索）
        if best_score > 0.0:  # 如果找到了改善
            prev_score = 0.0
            iteration = 0
            max_iterations = 2  # 最多迭代2次
            
            while iteration < max_iterations and best_score > prev_score + 0.001:
                prev_score = best_score
                prev_angle = best_angle_final
                prev_offset_x = best_offset_x_final
                prev_offset_y = best_offset_y_final
                
                # 在當前最佳值附近進行更細的搜索
                ultra_angle_range = np.arange(-1.0, 1.1, 0.1)  # ±1度，每0.1度
                ultra_offset_range = range(-5, 6, 1)  # ±5像素，每1像素
                
                for angle_offset in ultra_angle_range:
                    angle = prev_angle + angle_offset
                    # 限制在-80到+80度範圍內
                    if angle < -80 or angle > 80:
                        continue
                    
                    for dx in ultra_offset_range:
                        for dy in ultra_offset_range:
                            offset_x = prev_offset_x + dx
                            offset_y = prev_offset_y + dy
                            
                            if abs(offset_x) > max_translation or abs(offset_y) > max_translation:
                                continue
                            
                            h2, w2 = img2.shape
                            center = (w2 // 2, h2 // 2)
                            
                            M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                            img2_rot = cv2.warpAffine(img2, M_rot, (w2, h2),
                                                     borderMode=cv2.BORDER_CONSTANT,
                                                     borderValue=255)
                            
                            M_trans = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                            img2_final = cv2.warpAffine(img2_rot, M_trans, (w2, h2),
                                                       borderMode=cv2.BORDER_CONSTANT,
                                                       borderValue=255)
                            
                            img2_small = cv2.resize(img2_final, (small_w, small_h), 
                                                   interpolation=cv2.INTER_AREA)
                            score = self._fast_rotation_match(img1_small, img2_small)
                            
                            if score > best_score:
                                best_score = score
                                best_angle_final = angle
                                best_offset_x_final = offset_x
                                best_offset_y_final = offset_y
                            
                            if best_score > 0.98:
                                break
                        
                        if best_score > 0.98:
                            break
                    
                    if best_score > 0.98:
                        break
                
                iteration += 1
                
                # 如果沒有改善，提前退出
                if best_score <= prev_score:
                    break
        
        # 應用最佳變換到原始圖像
        h2, w2 = img2.shape
        center = (w2 // 2, h2 // 2)
        
        # 先旋轉
        M_rot = cv2.getRotationMatrix2D(center, best_angle_final, 1.0)
        img2_corrected = cv2.warpAffine(img2, M_rot, (w2, h2),
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=255)
        
        # 再平移
        M_trans = np.float32([[1, 0, best_offset_x_final], [0, 1, best_offset_y_final]])
        img2_corrected = cv2.warpAffine(img2_corrected, M_trans, (w2, h2),
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=255)
        
        return best_angle_final, (best_offset_x_final, best_offset_y_final), img2_corrected
    
    def _calculate_circularity(self, contour: np.ndarray) -> float:
        """
        計算輪廓的圓度（0-1，1表示完美圓形）
        
        Args:
            contour: 輪廓
            
        Returns:
            圓度值 (0-1)
        """
        area = cv2.contourArea(contour)
        if area == 0:
            return 0.0
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0
        
        # 圓度 = 4π * 面積 / 周長²
        circularity = 4 * np.pi * area / (perimeter ** 2)
        return min(1.0, circularity)
    
    def _detect_seal_shape(self, image: np.ndarray, bbox: Optional[Dict[str, int]] = None) -> str:
        """
        檢測印鑑形狀（方形或圓形）
        
        Args:
            image: 圖像
            bbox: 可選的邊界框，如果提供則只檢測該區域
            
        Returns:
            'square' 或 'circle'
        """
        # 轉換為灰度圖
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 如果提供了bbox，裁剪圖像
        if bbox:
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            h_img, w_img = gray.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            if w > 0 and h > 0:
                gray = gray[y:y+h, x:x+w]
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 查找輪廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # 默認返回方形
            return 'square'
        
        # 找到最大輪廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 計算圓度
        circularity = self._calculate_circularity(largest_contour)
        
        # 圓度 > 0.7 視為圓形，否則視為方形
        return 'circle' if circularity > 0.7 else 'square'
    
    def _find_square_orientation(self, image: np.ndarray) -> float:
        """
        找到方形印鑑的旋轉角度，使邊緣水平/垂直
        
        Args:
            image: 圖像
            
        Returns:
            旋轉角度（度）
        """
        # 轉換為灰度圖
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 邊緣檢測
        edges = cv2.Canny(gray, 50, 150)
        
        # 使用Hough線變換找到主要邊緣線
        # 限制搜索範圍以避免長時間運行
        min_line_length = max(10, int(min(gray.shape) * 0.1))  # 至少10像素，最多圖像尺寸的10%
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=min(50, int(min(gray.shape) * 0.1)), 
                                minLineLength=min_line_length, 
                                maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return 0.0
        
        # 計算所有線段的角度
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                # 將角度轉換到 -45 到 45 度範圍
                if angle > 45:
                    angle -= 90
                elif angle < -45:
                    angle += 90
                angles.append(angle)
        
        if not angles:
            return 0.0
        
        # 計算平均角度
        avg_angle = np.mean(angles)
        
        # 找到最接近0°、90°、180°或270°的角度
        # 由於我們已經將角度轉換到-45到45度範圍，只需要找到最接近0度的角度
        return avg_angle
    
    def _find_text_orientation(self, image: np.ndarray) -> float:
        """
        找到圓形印鑑的文字方向，使字體正視
        
        Args:
            image: 圖像
            
        Returns:
            旋轉角度（度）
        """
        # 轉換為灰度圖
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 使用形態學操作連接文字
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 查找輪廓（文字區域）
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # 找到所有文字區域的最小外接矩形
        weighted_angles = []
        weights = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:  # 過小的區域忽略
                continue
            
            # 最小外接矩形
            rect = cv2.minAreaRect(contour)
            angle = rect[2]
            
            # 調整角度範圍到 -45 到 45 度
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            
            weighted_angles.append(angle)
            weights.append(area)
        
        if not weighted_angles:
            return 0.0
        
        # 加權平均
        avg_angle = np.average(weighted_angles, weights=weights)
        
        return avg_angle
    
    def _align_image1(self, image: np.ndarray, bbox: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        對齊圖像1：平移（印鑑中心對齊到圖片中心）+ 旋轉（根據印鑑形狀）
        
        Args:
            image: 原始圖像（已裁切）
            bbox: 可選的邊界框信息（用於計算印鑑中心）
            
        Returns:
            (對齊後的圖像, 旋轉角度, (平移x, 平移y))
        """
        h, w = image.shape[:2]
        
        # 步驟1：平移對齊（印鑑中心對齊到圖片中心）
        translation_x = 0
        translation_y = 0
        
        if bbox:
            # 從bbox計算印鑑中心（相對於裁切後的圖像）
            # 注意：bbox是相對於原始圖像的，但image已經是裁切後的
            # 所以需要調整bbox坐標
            seal_center_x = bbox['x'] + bbox['width'] / 2
            seal_center_y = bbox['y'] + bbox['height'] / 2
            
            # 圖片中心
            image_center_x = w / 2
            image_center_y = h / 2
            
            # 計算偏移（注意：如果image已經是裁切後的，需要調整）
            # 實際上，如果image已經是裁切後的，bbox的坐標需要相對於裁切後的圖像
            # 但為了簡化，我們假設bbox['x']和bbox['y']相對於裁切後的圖像為0
            # 或者我們直接使用bbox的中心相對於裁切後圖像的位置
            seal_center_x_local = bbox['width'] / 2
            seal_center_y_local = bbox['height'] / 2
            
            translation_x = image_center_x - seal_center_x_local
            translation_y = image_center_y - seal_center_y_local
        else:
            # 如果沒有bbox，嘗試從圖像中檢測印鑑中心
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    seal_center_x_local = M["m10"] / M["m00"]
                    seal_center_y_local = M["m01"] / M["m00"]
                    translation_x = w / 2 - seal_center_x_local
                    translation_y = h / 2 - seal_center_y_local
        
        # 應用平移
        if abs(translation_x) > 0.5 or abs(translation_y) > 0.5:
            M_trans = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
            image = cv2.warpAffine(image, M_trans, (w, h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255, 255, 255))
        
        # 步驟2：旋轉對齊（根據印鑑形狀）
        # 添加異常處理，避免卡住
        try:
            shape = self._detect_seal_shape(image, bbox=None)  # image已經是裁切後的，不需要bbox
            
            if shape == 'square':
                rotation_angle = self._find_square_orientation(image)
            else:  # circle
                rotation_angle = self._find_text_orientation(image)
        except Exception as e:
            # 如果檢測失敗，使用默認值（不旋轉）
            print(f"警告：印鑑形狀檢測或旋轉角度計算失敗: {e}")
            rotation_angle = 0.0
        
        # 應用旋轉
        if abs(rotation_angle) > 0.1:
            center = (w / 2, h / 2)
            M_rot = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            image = cv2.warpAffine(image, M_rot, (w, h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255, 255, 255))
        else:
            rotation_angle = 0.0
        
        return image, rotation_angle, (int(translation_x), int(translation_y))
    
    def _align_image2(self, image: np.ndarray, bbox: Optional[Dict[str, int]] = None) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        對齊圖像2：平移（印鑑中心對齊到圖片中心）
        注意：旋轉將在後續步驟中相對於圖像1進行
        
        Args:
            image: 原始圖像（已裁切）
            bbox: 可選的邊界框信息（用於計算印鑑中心）
            
        Returns:
            (對齊後的圖像, (平移x, 平移y))
        """
        h, w = image.shape[:2]
        
        # 平移對齊（印鑑中心對齊到圖片中心）
        translation_x = 0
        translation_y = 0
        
        if bbox:
            seal_center_x_local = bbox['width'] / 2
            seal_center_y_local = bbox['height'] / 2
            
            image_center_x = w / 2
            image_center_y = h / 2
            
            translation_x = image_center_x - seal_center_x_local
            translation_y = image_center_y - seal_center_y_local
        else:
            # 如果沒有bbox，嘗試從圖像中檢測印鑑中心
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    seal_center_x_local = M["m10"] / M["m00"]
                    seal_center_y_local = M["m01"] / M["m00"]
                    translation_x = w / 2 - seal_center_x_local
                    translation_y = h / 2 - seal_center_y_local
        
        # 應用平移
        if abs(translation_x) > 0.5 or abs(translation_y) > 0.5:
            M_trans = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
            image = cv2.warpAffine(image, M_trans, (w, h),
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255, 255, 255))
        
        return image, (int(translation_x), int(translation_y))
    
    def _align_image2_to_image1(
        self, 
        image1: np.ndarray, 
        image2: np.ndarray,
        rotation_range: float = 45.0,
        translation_range: int = 100
    ) -> Tuple[np.ndarray, float, Tuple[int, int], float, Dict[str, float]]:
        """
        將圖像2對齊到圖像1，使用相似度計算優化對齊參數
        
        Args:
            image1: 參考圖像（已去背景）
            image2: 待對齊圖像（已去背景）
            rotation_range: 旋轉角度搜索範圍（度）
            translation_range: 平移偏移搜索範圍（像素）
            
        Returns:
            (對齊後的圖像2, 最佳旋轉角度, (最佳x偏移, 最佳y偏移), 最佳相似度, 詳細指標)
        """
        # 轉換為灰度圖（如果輸入是彩色圖像）
        if len(image1.shape) == 3:
            img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = image1.copy()
        
        if len(image2.shape) == 3:
            img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            img2_gray = image2.copy()
        
        h2, w2 = img2_gray.shape
        
        # 階段1：快速粗搜索
        # 使用縮小圖像進行快速評估
        scale_factor = 0.3
        small_h = int(h2 * scale_factor)
        small_w = int(w2 * scale_factor)
        img1_small = cv2.resize(img1_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        img2_small = cv2.resize(img2_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        # 粗搜索參數
        angle_step_coarse = 5.0  # 每5度
        offset_step_coarse = 10  # 每10像素（完整尺寸）
        offset_step_small = int(offset_step_coarse * scale_factor)  # 縮小圖像的步長
        
        candidates = []  # 存儲候選 (angle, offset_x, offset_y, score)
        
        # 旋轉角度範圍：-rotation_range 到 +rotation_range
        for angle in np.arange(-rotation_range, rotation_range + angle_step_coarse, angle_step_coarse):
            # 限制在合理範圍內
            if angle < -80 or angle > 80:
                continue
            
            # 平移偏移範圍：-translation_range 到 +translation_range（完整尺寸）
            for dx_full in range(-translation_range, translation_range + offset_step_coarse, offset_step_coarse):
                for dy_full in range(-translation_range, translation_range + offset_step_coarse, offset_step_coarse):
                    # 在縮小圖像上應用變換（平移偏移需要按比例縮小）
                    dx_small = int(dx_full * scale_factor)
                    dy_small = int(dy_full * scale_factor)
                    
                    center_small = (small_w // 2, small_h // 2)
                    M_rot = cv2.getRotationMatrix2D(center_small, angle, 1.0)
                    img2_rot_small = cv2.warpAffine(img2_small, M_rot, (small_w, small_h),
                                                   borderMode=cv2.BORDER_CONSTANT,
                                                   borderValue=255)
                    
                    M_trans = np.float32([[1, 0, dx_small], [0, 1, dy_small]])
                    img2_trans_small = cv2.warpAffine(img2_rot_small, M_trans, (small_w, small_h),
                                                     borderMode=cv2.BORDER_CONSTANT,
                                                     borderValue=255)
                    
                    # 使用快速旋轉匹配進行評估
                    score = self._fast_rotation_match(img1_small, img2_trans_small)
                    candidates.append((angle, dx_full, dy_full, score))
        
        # 按分數排序，選擇前5個最佳候選
        if not candidates:
            # 如果沒有候選，使用默認值（不變換）
            best_angle = 0.0
            best_offset_x = 0
            best_offset_y = 0
            best_similarity = 0.0
            best_metrics = {}
        else:
            candidates.sort(key=lambda x: x[3], reverse=True)
            top_candidates = candidates[:5]
            
            # 階段2：完整相似度計算
            best_angle = 0.0
            best_offset_x = 0
            best_offset_y = 0
            best_similarity = 0.0
            best_metrics = {}
            
            for angle, offset_x, offset_y, _ in top_candidates:
                # 應用旋轉和平移到完整尺寸圖像
                center = (w2 // 2, h2 // 2)
                M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                img2_rot = cv2.warpAffine(img2_gray, M_rot, (w2, h2),
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=255)
                
                M_trans = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                img2_transformed = cv2.warpAffine(img2_rot, M_trans, (w2, h2),
                                                 borderMode=cv2.BORDER_CONSTANT,
                                                 borderValue=255)
                
                # 確保尺寸相同
                h1, w1 = img1_gray.shape
                if h1 != h2 or w1 != w2:
                    target_h = max(h1, h2)
                    target_w = max(w1, w2)
                    img1_resized = cv2.resize(img1_gray, (target_w, target_h))
                    img2_resized = cv2.resize(img2_transformed, (target_w, target_h))
                else:
                    img1_resized = img1_gray
                    img2_resized = img2_transformed
                
                # 使用快速旋轉匹配計算相似度
                similarity = self._fast_rotation_match(img1_resized, img2_resized)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_angle = angle
                    best_offset_x = offset_x
                    best_offset_y = offset_y
                    best_metrics = {}  # 不再使用 enhanced_metrics
        
        # 階段3：細搜索優化（在最佳候選附近）
        if best_similarity > 0.0:
            fine_angle_range = np.arange(-2.0, 2.1, 0.5)  # ±2度，每0.5度
            fine_offset_range = range(-10, 11, 1)  # ±10像素，每1像素
            
            for angle_offset in fine_angle_range:
                angle = best_angle + angle_offset
                if angle < -80 or angle > 80:
                    continue
                
                for dx in fine_offset_range:
                    for dy in fine_offset_range:
                        offset_x = best_offset_x + dx
                        offset_y = best_offset_y + dy
                        
                        if abs(offset_x) > translation_range or abs(offset_y) > translation_range:
                            continue
                        
                        # 應用變換
                        center = (w2 // 2, h2 // 2)
                        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                        img2_rot = cv2.warpAffine(img2_gray, M_rot, (w2, h2),
                                                 borderMode=cv2.BORDER_CONSTANT,
                                                 borderValue=255)
                        
                        M_trans = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                        img2_transformed = cv2.warpAffine(img2_rot, M_trans, (w2, h2),
                                                          borderMode=cv2.BORDER_CONSTANT,
                                                          borderValue=255)
                        
                        # 確保尺寸相同
                        h1, w1 = img1_gray.shape
                        if h1 != h2 or w1 != w2:
                            target_h = max(h1, h2)
                            target_w = max(w1, w2)
                            img1_resized = cv2.resize(img1_gray, (target_w, target_h))
                            img2_resized = cv2.resize(img2_transformed, (target_w, target_h))
                        else:
                            img1_resized = img1_gray
                            img2_resized = img2_transformed
                        
                        # 使用快速旋轉匹配計算相似度
                        similarity = self._fast_rotation_match(img1_resized, img2_resized)
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_angle = angle
                            best_offset_x = offset_x
                            best_offset_y = offset_y
                            best_metrics = {}  # 不再使用 enhanced_metrics
        
        # 應用最佳變換到原始圖像（保持原始顏色）
        if len(image2.shape) == 3:
            img2_final = image2.copy()
        else:
            img2_final = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
        
        center = (w2 // 2, h2 // 2)
        M_rot = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        img2_rotated = cv2.warpAffine(img2_final, M_rot, (w2, h2),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
        
        M_trans = np.float32([[1, 0, best_offset_x], [0, 1, best_offset_y]])
        img2_aligned = cv2.warpAffine(img2_rotated, M_trans, (w2, h2),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
        
        return img2_aligned, best_angle, (best_offset_x, best_offset_y), best_similarity, best_metrics
    
    def _fast_rotation_match(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        快速旋轉匹配評估（用於角度搜索）- 改進版
        
        使用多種方法組合，提高準確度：
        1. 邊緣匹配（Canny + 模板匹配）- 對旋轉最敏感
        2. SSIM相似度 - 結構相似性
        3. 像素相似度 - 快速評估
        4. 輪廓匹配 - 僅對小圖像使用
        
        Args:
            img1: 參考圖像
            img2: 待比對圖像
            
        Returns:
            匹配分數 (0-1)
        """
        # 確保尺寸相同（如果不同，調整到相同尺寸）
        if img1.shape != img2.shape:
            target_h = max(img1.shape[0], img2.shape[0])
            target_w = max(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # 方法1：邊緣檢測 + 模板匹配（最快且對旋轉敏感）
        edges1 = cv2.Canny(img1, 50, 150)
        edges2 = cv2.Canny(img2, 50, 150)
        
        edge_match = 0.0
        if np.sum(edges1) > 0 and np.sum(edges2) > 0:
            # 使用模板匹配
            result = cv2.matchTemplate(edges1, edges2, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            edge_match = float(max_val)
        elif np.sum(edges1) == 0 and np.sum(edges2) == 0:
            # 如果都沒有邊緣，可能是空白圖像
            edge_match = 0.5
        
        # 方法2：簡化版SSIM（結構相似性）
        img1_float = img1.astype(np.float64)
        img2_float = img2.astype(np.float64)
        
        mu1 = cv2.GaussianBlur(img1_float, (5, 5), 1.0)
        mu2 = cv2.GaussianBlur(img2_float, (5, 5), 1.0)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1_float ** 2, (5, 5), 1.0) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2_float ** 2, (5, 5), 1.0) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1_float * img2_float, (5, 5), 1.0) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_score = float(np.mean(ssim_map))
        
        # 方法3：像素相似度（快速，但對旋轉不敏感）
        mask1 = img1 < 250  # 非背景像素
        mask2 = img2 < 250
        mask_combined = mask1 & mask2
        
        pixel_similarity = 0.0
        if np.sum(mask_combined) > 0:
            diff = cv2.absdiff(img1, img2)
            pixel_similarity = 1.0 - (np.sum(diff[mask_combined] > 10) / np.sum(mask_combined))
        
        # 方法4：輪廓匹配（僅對小圖像使用）
        contour_match = 0.0
        try:
            img_size = img1.shape[0] * img1.shape[1]
            if img_size < 40000:  # 只對小圖像使用（約200x200以下）
                contours1, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours2, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours1) > 0 and len(contours2) > 0:
                    c1 = max(contours1, key=cv2.contourArea)
                    c2 = max(contours2, key=cv2.contourArea)
                    if cv2.contourArea(c1) > 10 and cv2.contourArea(c2) > 10:
                        match = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I2, 0)
                        contour_match = 1.0 / (1.0 + match * 10)
        except Exception:
            contour_match = 0.0
        
        # 加權組合（改進權重分配）
        if contour_match > 0:
            score = edge_match * 0.4 + ssim_score * 0.35 + pixel_similarity * 0.15 + contour_match * 0.1
        else:
            score = edge_match * 0.45 + ssim_score * 0.4 + pixel_similarity * 0.15
        
        return max(0.0, min(1.0, score))  # 確保分數在0-1範圍內
    
    def _compute_image_hash(self, image: np.ndarray) -> str:
        """
        計算圖像的哈希值（用於快速檢測相同圖像）
        
        Args:
            image: 圖像陣列
            
        Returns:
            圖像哈希值字符串
        """
        # 將圖像縮小到固定尺寸以加快計算
        small = cv2.resize(image, (32, 32))
        # 計算哈希
        image_bytes = small.tobytes()
        return hashlib.md5(image_bytes).hexdigest()
    
    def _check_identical_images(self, img1: np.ndarray, img2: np.ndarray, 
                                bbox1: Optional[Dict[str, int]] = None,
                                bbox2: Optional[Dict[str, int]] = None) -> Optional[float]:
        """
        快速檢測兩個圖像是否完全相同（在裁切前）
        
        Args:
            img1: 第一個圖像
            img2: 第二個圖像
            bbox1: 圖像1的裁切區域
            bbox2: 圖像2的裁切區域
            
        Returns:
            如果完全相同返回 1.0，否則返回 None
        """
        # 如果兩個圖像尺寸相同，直接比較
        if img1.shape == img2.shape:
            # 使用更寬鬆的比較：允許微小的差異（由於 JPEG 壓縮等）
            diff = cv2.absdiff(img1, img2)
            diff_pixels = np.count_nonzero(diff > 1)  # 允許1像素的差異
            total_pixels = img1.size
            if diff_pixels / total_pixels < 0.001:  # 差異小於0.1%視為相同
                return 1.0
        
        # 如果提供了裁切區域，比較裁切後的圖像
        if bbox1 and bbox2:
            # 即使裁切區域不完全相同，如果非常接近，也進行比較
            bbox1_area = bbox1['width'] * bbox1['height']
            bbox2_area = bbox2['width'] * bbox2['height']
            area_ratio = min(bbox1_area, bbox2_area) / max(bbox1_area, bbox2_area) if max(bbox1_area, bbox2_area) > 0 else 0
            
            # 如果面積比例很高（>0.95），且位置接近，進行比較
            if area_ratio > 0.95:
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                
                x1, y1, w1_bbox, h1_bbox = bbox1['x'], bbox1['y'], bbox1['width'], bbox1['height']
                x2, y2, w2_bbox, h2_bbox = bbox2['x'], bbox2['y'], bbox2['width'], bbox2['height']
                
                # 確保裁切區域在圖像範圍內
                x1 = max(0, min(x1, w1 - 1))
                y1 = max(0, min(y1, h1 - 1))
                w1_bbox = min(w1_bbox, w1 - x1)
                h1_bbox = min(h1_bbox, h1 - y1)
                
                x2 = max(0, min(x2, w2 - 1))
                y2 = max(0, min(y2, h2 - 1))
                w2_bbox = min(w2_bbox, w2 - x2)
                h2_bbox = min(h2_bbox, h2 - y2)
                
                if w1_bbox > 0 and h1_bbox > 0 and w2_bbox > 0 and h2_bbox > 0:
                    crop1 = img1[y1:y1+h1_bbox, x1:x1+w1_bbox]
                    crop2 = img2[y2:y2+h2_bbox, x2:x2+w2_bbox]
                    
                    # 調整到相同尺寸進行比較
                    if crop1.shape != crop2.shape:
                        target_h = max(crop1.shape[0], crop2.shape[0])
                        target_w = max(crop1.shape[1], crop2.shape[1])
                        crop1 = cv2.resize(crop1, (target_w, target_h))
                        crop2 = cv2.resize(crop2, (target_w, target_h))
                    
                    # 使用寬鬆的比較
                    diff = cv2.absdiff(crop1, crop2)
                    diff_pixels = np.count_nonzero(diff > 1)
                    total_pixels = crop1.size
                    if diff_pixels / total_pixels < 0.001:  # 差異小於0.1%視為相同
                        return 1.0
        
        return None
    
    def _adaptive_edge_detection(self, img: np.ndarray) -> np.ndarray:
        """
        自適應邊緣檢測，根據圖像特性自動調整 Canny 參數
        
        Args:
            img: 輸入圖像
            
        Returns:
            邊緣檢測結果
        """
        # 計算圖像統計信息
        mean_val = np.mean(img)
        std_val = np.std(img)
        
        # 根據統計信息調整 Canny 參數
        # 對於低對比度圖像，使用較低的閾值
        # 對於高對比度圖像，使用較高的閾值
        
        # 嘗試多種參數組合
        param_combinations = [
            (int(mean_val * 0.5), int(mean_val * 1.5)),  # 基於均值
            (int(std_val * 0.3), int(std_val * 0.6)),     # 基於標準差
            (30, 100),  # 較低閾值，適合低對比度
            (50, 150),  # 默認參數
            (70, 200),  # 較高閾值，適合高對比度
            (20, 80),   # 更寬鬆的參數
        ]
        
        best_edges = None
        best_edge_count = 0
        target_edge_ratio = 0.05  # 目標邊緣像素比例（5%）
        total_pixels = img.size
        
        for low, high in param_combinations:
            # 確保參數在合理範圍內
            low = max(10, min(255, low))
            high = max(low + 10, min(255, high))
            
            try:
                edges = cv2.Canny(img, low, high)
                edge_count = np.count_nonzero(edges)
                edge_ratio = edge_count / total_pixels
                
                # 選擇邊緣數量最接近目標比例的結果
                if best_edges is None or abs(edge_ratio - target_edge_ratio) < abs(best_edge_count / total_pixels - target_edge_ratio):
                    best_edges = edges
                    best_edge_count = edge_count
                    
                    # 如果已經很接近目標，可以提前退出
                    if abs(edge_ratio - target_edge_ratio) < 0.01:
                        break
            except:
                continue
        
        # 如果所有參數都失敗，使用默認參數
        if best_edges is None:
            best_edges = cv2.Canny(img, 50, 150)
        
        return best_edges
    
    def _calculate_edge_similarity_advanced(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        計算改進的邊緣相似度，使用自適應邊緣檢測和多種方法
        
        Args:
            img1: 第一個圖像
            img2: 第二個圖像
            
        Returns:
            邊緣相似度 (0-1)
        """
        # 使用自適應邊緣檢測
        edges1 = self._adaptive_edge_detection(img1)
        edges2 = self._adaptive_edge_detection(img2)
        
        # 方法1: 邊緣位置匹配（交集/並集）
        edge_intersection = np.sum((edges1 > 0) & (edges2 > 0))
        edge_union = np.sum((edges1 > 0) | (edges2 > 0))
        position_similarity = edge_intersection / edge_union if edge_union > 0 else 0.0
        
        # 方法2: 邊緣密度相似度
        edge1_density = np.count_nonzero(edges1) / edges1.size
        edge2_density = np.count_nonzero(edges2) / edges2.size
        density_similarity = 1.0 - abs(edge1_density - edge2_density) / max(edge1_density + edge2_density, 0.001)
        
        # 方法3: 使用距離變換計算邊緣距離相似度
        try:
            dist1 = cv2.distanceTransform(255 - edges1, cv2.DIST_L2, 5)
            dist2 = cv2.distanceTransform(255 - edges2, cv2.DIST_L2, 5)
            # 計算距離圖的相似度
            dist_diff = np.abs(dist1 - dist2)
            max_dist = max(np.max(dist1), np.max(dist2))
            distance_similarity = 1.0 - (np.mean(dist_diff) / max_dist) if max_dist > 0 else 0.0
            distance_similarity = max(0.0, min(1.0, distance_similarity))
        except:
            distance_similarity = 0.0
        
        # 綜合三種方法（加權平均）
        edge_similarity = (
            position_similarity * 0.5 +
            density_similarity * 0.3 +
            distance_similarity * 0.2
        )
        
        return float(max(0.0, min(1.0, edge_similarity)))
    
    def _calculate_enhanced_similarity(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
        """
        計算增強的相似度指標（使用多種方法）
        
        Args:
            img1: 第一個圖像
            img2: 第二個圖像
            
        Returns:
            包含各種相似度指標的字典
        """
        # 確保尺寸相同
        if img1.shape != img2.shape:
            target_h = max(img1.shape[0], img2.shape[0])
            target_w = max(img1.shape[1], img2.shape[1])
            img1 = cv2.resize(img1, (target_w, target_h))
            img2 = cv2.resize(img2, (target_w, target_h))
        
        metrics = {}
        
        # 1. SSIM
        metrics['ssim'] = self._calculate_ssim(img1, img2)
        
        # 2. 模板匹配
        metrics['template_match'] = self._template_match(img1, img2)
        
        # 3. 像素差異（轉換為相似度）
        pixel_diff = self._pixel_difference(img1, img2)
        metrics['pixel_similarity'] = 1.0 - pixel_diff
        
        # 4. 直方圖相似度
        hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
        hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        metrics['histogram_similarity'] = float(hist_corr)
        
        # 5. 改進的邊緣相似度（使用自適應邊緣檢測）
        edge_similarity = self._calculate_edge_similarity_advanced(img1, img2)
        metrics['edge_similarity'] = float(edge_similarity)
        
        # 6. 精確像素匹配率（對於相同圖像很重要）
        exact_match = np.sum(img1 == img2) / img1.size
        metrics['exact_match_ratio'] = float(exact_match)
        
        # 7. 均方誤差（MSE）轉換為相似度
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
        max_mse = 255.0 ** 2
        mse_similarity = 1.0 - (mse / max_mse)
        metrics['mse_similarity'] = float(max(0.0, mse_similarity))
        
        return metrics
    
    def _calculate_final_similarity(self, enhanced_metrics: Dict[str, float]) -> float:
        """
        根據增強相似度指標計算最終相似度（加權計算）
        
        Args:
            enhanced_metrics: 增強相似度指標字典，包含：
                - ssim: SSIM 結構相似性
                - template_match: 模板匹配相似度
                - pixel_similarity: 像素相似度
                - histogram_similarity: 直方圖相似度
                - edge_similarity: 邊緣相似度
                - exact_match_ratio: 精確匹配率
                - mse_similarity: MSE 相似度
                
        Returns:
            最終相似度 (0-1)
        """
        similarity_ssim = enhanced_metrics['ssim']
        similarity_template = enhanced_metrics['template_match']
        pixel_similarity = enhanced_metrics['pixel_similarity']
        histogram_similarity = enhanced_metrics['histogram_similarity']
        edge_similarity = enhanced_metrics['edge_similarity']
        exact_match_ratio = enhanced_metrics['exact_match_ratio']
        mse_similarity = enhanced_metrics['mse_similarity']
        
        # 如果精確匹配率很高，直接使用高相似度
        if exact_match_ratio > 0.95:
            # 對於幾乎完全相同的圖像，使用更高的相似度
            # 將 0.95-1.0 映射到 0.98-1.0，確保高匹配率得到高相似度
            similarity = 0.98 + (exact_match_ratio - 0.95) * 0.4  # 0.95 -> 0.98, 1.0 -> 1.0
        elif exact_match_ratio > 0.90:
            # 對於高匹配率（0.90-0.95），使用加權計算但偏向精確匹配率
            base_similarity = (
                similarity_ssim * 0.20 +
                similarity_template * 0.15 +
                pixel_similarity * 0.15 +
                histogram_similarity * 0.20 +  # 提高直方圖相似度權重
                edge_similarity * 0.05 +  # 降低邊緣相似度權重
                exact_match_ratio * 0.20 +
                mse_similarity * 0.05
            )
            # 進一步提升：精確匹配率越高，最終相似度越高
            similarity = max(base_similarity, exact_match_ratio * 0.95)
        else:
            # 綜合相似度（改進的加權平均，考慮更多指標）
            # 調整權重：提高直方圖相似度，降低邊緣相似度
            similarity = (
                similarity_ssim * 0.25 +
                similarity_template * 0.20 +
                pixel_similarity * 0.20 +
                histogram_similarity * 0.20 +  # 從 0.10 提高到 0.20
                edge_similarity * 0.05 +  # 從 0.10 降低到 0.05
                exact_match_ratio * 0.05 +  # 從 0.10 降低到 0.05（為直方圖讓路）
                mse_similarity * 0.05
            )
        
        # 如果精確匹配率很高，進一步提升相似度（額外保護）
        if exact_match_ratio > 0.90:
            similarity = max(similarity, exact_match_ratio * 0.98)
        
        # 特殊處理：當直方圖相似度極高時（說明是同一印章但掃描品質不同）
        # 如果直方圖相似度 > 0.95 且 SSIM > 0.60，即使邊緣相似度低，也提升相似度
        if histogram_similarity > 0.95 and similarity_ssim > 0.60:
            # 結構相似性優先模式：降低對邊緣和精確匹配的依賴
            structure_based_similarity = (
                similarity_ssim * 0.35 +
                histogram_similarity * 0.35 +
                pixel_similarity * 0.20 +
                similarity_template * 0.10
            )
            # 如果結構相似性計算出的相似度更高，使用它
            similarity = max(similarity, structure_based_similarity)
            # 進一步提升：直方圖相似度極高時，額外提升
            if histogram_similarity > 0.98:
                similarity = max(similarity, 0.90)  # 至少提升到 0.90
        
        # 對於相同圖像的特殊處理：如果多個指標都顯示極高相似度，強制提升
        if (exact_match_ratio > 0.85 and 
            pixel_similarity > 0.90 and 
            similarity_ssim > 0.90 and 
            similarity_template > 0.90):
            # 多個指標都顯示極高相似度，很可能是同一張圖像
            similarity = max(similarity, 0.95)
        
        # 額外處理：如果直方圖相似度極高（>0.95），進一步提升
        if histogram_similarity > 0.95:
            similarity = max(similarity, histogram_similarity * 0.95)
        
        return max(0.0, min(1.0, similarity))  # 確保在 0-1 範圍內
    
    def compare_images(self, image1: np.ndarray, image2: np.ndarray, 
                      enable_rotation_search: bool = True,
                      enable_translation_search: bool = True,  # 預設開啟，因為人工標記印鑑無法確保中心點都一致
                      bbox1: Optional[Dict[str, int]] = None,
                      bbox2: Optional[Dict[str, int]] = None) -> Tuple[bool, float, dict, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        比對兩個圖像
        
        Args:
            image1: 第一個圖像
            image2: 第二個圖像
            enable_rotation_search: 是否啟用旋轉角度搜索（預設為 True）
            enable_translation_search: 是否啟用平移搜索（預設為 True，因為人工標記印鑑無法確保中心點都一致）
            bbox1: 圖像1的裁切區域 {"x": int, "y": int, "width": int, "height": int}
            bbox2: 圖像2的裁切區域 {"x": int, "y": int, "width": int, "height": int}
            
        Returns:
            (是否一致, 相似度, 詳細資訊, 校正後的圖像2, 校正後的圖像1)
        """
        # 輸入驗證
        if image1 is None or image2 is None:
            raise ValueError("圖像不能為 None")
        
        if image1.size == 0 or image2.size == 0:
            raise ValueError("圖像不能為空")
        
        # 圖像已經在 service 層完成裁切、去背景和對齊處理
        # 轉換為灰度圖（如果輸入是彩色圖像），因為相似度計算方法需要灰度圖
        if len(image1.shape) == 3:
            img1_processed = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            img1_processed = image1.copy()
        
        if len(image2.shape) == 3:
            img2_processed = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            img2_processed = image2.copy()
        
        # 確保兩個圖像尺寸相同
        h1, w1 = img1_processed.shape
        h2, w2 = img2_processed.shape
        
        if h1 != h2 or w1 != w2:
            # 調整到相同尺寸
            target_h = max(h1, h2)
            target_w = max(w1, w2)
            img1_final = cv2.resize(img1_processed, (target_w, target_h))
            img2_final = cv2.resize(img2_processed, (target_w, target_h))
        else:
            img1_final = img1_processed
            img2_final = img2_processed
        
        # 直接計算相似度指標（不使用 _calculate_enhanced_similarity）
        similarity_ssim = self._calculate_ssim(img1_final, img2_final)
        similarity_template = self._template_match(img1_final, img2_final)
        pixel_diff = self._pixel_difference(img1_final, img2_final)
        pixel_similarity = 1.0 - pixel_diff
        
        # 計算直方圖相似度
        hist1 = cv2.calcHist([img1_final], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([img2_final], [0], None, [256], [0, 256])
        histogram_similarity = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
        
        # 計算邊緣相似度
        edge_similarity = self._calculate_edge_similarity_advanced(img1_final, img2_final)
        
        # 計算精確匹配率
        exact_match_ratio = float(np.sum(img1_final == img2_final) / img1_final.size)
        
        # 計算MSE相似度
        mse = np.mean((img1_final.astype(np.float64) - img2_final.astype(np.float64)) ** 2)
        max_mse = 255.0 ** 2
        mse_similarity = float(max(0.0, 1.0 - (mse / max_mse)))
        
        # 使用簡單的加權組合計算最終相似度
        similarity = (
            similarity_ssim * 0.4 +
            similarity_template * 0.3 +
            pixel_similarity * 0.2 +
            histogram_similarity * 0.1
        )
        
        # 確保相似度在合理範圍內
        similarity = min(1.0, max(0.0, similarity))
        
        # 判斷是否一致
        is_match = similarity >= self.threshold
        
        # 計算尺寸資訊（使用輸入圖像的尺寸）
        h1_orig, w1_orig = image1.shape[:2]
        h2_orig, w2_orig = image2.shape[:2]
        size_ratio = (w1_orig * h1_orig) / (w2_orig * h2_orig) if (w2_orig * h2_orig) > 0 else 1.0
        
        details = {
            'similarity': similarity,
            'ssim': similarity_ssim,
            'template_match': similarity_template,
            'pixel_diff': pixel_diff,
            'pixel_similarity': pixel_similarity,
            'histogram_similarity': round(histogram_similarity, 4),
            'edge_similarity': round(edge_similarity, 4),
            'exact_match_ratio': round(exact_match_ratio, 4),
            'mse_similarity': round(mse_similarity, 4),
            'threshold': self.threshold,
            'rotation_angle': 0.0,  # 圖像已對齊，不需要旋轉
            'translation_offset': None,  # 圖像已對齊，不需要平移
            'similarity_before_correction': None,  # 不再計算校正前相似度
            'improvement': None,  # 不再計算改善幅度
            'is_identical': False,
            'image1_size': (h1_orig, w1_orig),
            'image2_size': (h2_orig, w2_orig),
            'size_ratio': round(size_ratio, 4)
        }
        
        # 返回對齊後的圖像（輸入圖像已經是對齊後的）
        return is_match, similarity, details, image2.copy(), image1.copy()
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        計算結構相似性指數 (SSIM)
        
        Args:
            img1: 第一個圖像
            img2: 第二個圖像
            
        Returns:
            SSIM 值 (0-1)
        """
        # 簡化版 SSIM 計算
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(np.mean(ssim_map))
    
    def _template_match(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        改進的模板匹配，使用多種方法和多尺度匹配
        
        Args:
            img1: 第一個圖像
            img2: 第二個圖像
            
        Returns:
            匹配度 (0-1)
        """
        # 確保尺寸相同（模板匹配要求模板不能大於原圖）
        if img1.shape[0] > img2.shape[0] or img1.shape[1] > img2.shape[1]:
            # 如果 img1 更大，交換順序
            img1, img2 = img2, img1
        
        # 如果尺寸差異太大，先進行尺寸標準化
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if abs(h1 - h2) > max(h1, h2) * 0.1 or abs(w1 - w2) > max(w1, w2) * 0.1:
            # 尺寸差異超過10%，先標準化
            target_h = min(h1, h2)
            target_w = min(w1, w2)
            img1 = cv2.resize(img1, (target_w, target_h))
            img2 = cv2.resize(img2, (target_w, target_h))
        
        # 方法1: TM_CCOEFF_NORMED（相關性係數，歸一化）
        result1 = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        _, max_val1, _, _ = cv2.minMaxLoc(result1)
        
        # 方法2: TM_CCORR_NORMED（相關性，歸一化）
        result2 = cv2.matchTemplate(img1, img2, cv2.TM_CCORR_NORMED)
        _, max_val2, _, _ = cv2.minMaxLoc(result2)
        
        # 方法3: TM_SQDIFF_NORMED（平方差，歸一化，值越小越好）
        result3 = cv2.matchTemplate(img1, img2, cv2.TM_SQDIFF_NORMED)
        min_val3, _, _, _ = cv2.minMaxLoc(result3)
        max_val3 = 1.0 - min_val3  # 轉換為相似度（值越大越好）
        
        # 選擇最佳結果
        max_val = max(max_val1, max_val2, max_val3)
        
        # 多尺度匹配：如果圖像較大，嘗試縮小後匹配
        if min(h1, w1) > 200:
            scale_factor = 0.5
            small_h = int(h1 * scale_factor)
            small_w = int(w1 * scale_factor)
            img1_small = cv2.resize(img1, (small_w, small_h))
            img2_small = cv2.resize(img2, (small_w, small_h))
            
            result_small = cv2.matchTemplate(img1_small, img2_small, cv2.TM_CCOEFF_NORMED)
            _, max_val_small, _, _ = cv2.minMaxLoc(result_small)
            
            # 綜合原尺寸和多尺度的結果
            max_val = max(max_val, max_val_small * 0.95)  # 多尺度結果稍微降低權重
        
        return float(max_val)
    
    def _pixel_difference(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        計算像素差異率
        
        Args:
            img1: 第一個圖像
            img2: 第二個圖像
            
        Returns:
            差異率 (0-1)
        """
        diff = cv2.absdiff(img1, img2)
        diff_pixels = np.count_nonzero(diff)
        total_pixels = img1.size
        
        return diff_pixels / total_pixels
    
    def compare_files(self, image1_path: str, image2_path: str, 
                     enable_rotation_search: bool = True,
                     enable_translation_search: bool = True,  # 預設開啟，因為人工標記印鑑無法確保中心點都一致
                     bbox1: Optional[Dict[str, int]] = None,
                     bbox2: Optional[Dict[str, int]] = None) -> Tuple[bool, float, dict, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        比對兩個圖像文件
        
        Args:
            image1_path: 第一個圖像路徑
            image2_path: 第二個圖像路徑
            enable_rotation_search: 是否啟用旋轉角度搜索（預設為 True）
            enable_translation_search: 是否啟用平移搜索（預設為 True，因為人工標記印鑑無法確保中心點都一致）
            bbox1: 圖像1的裁切區域 {"x": int, "y": int, "width": int, "height": int}
            bbox2: 圖像2的裁切區域 {"x": int, "y": int, "width": int, "height": int}
            
        Returns:
            (是否一致, 相似度, 詳細資訊, 校正後的圖像2, 校正後的圖像1)
        """
        # 輸入驗證
        if not image1_path or not image2_path:
            raise ValueError("圖像路徑不能為空")
        
        img1 = self.load_image(image1_path)
        img2 = self.load_image(image2_path)
        
        if img1 is None or img2 is None:
            error_msg = f"無法載入圖像: image1={image1_path}, image2={image2_path}"
            return False, 0.0, {'error': error_msg}, None, None
        
        try:
            return self.compare_images(
                img1, img2, 
                enable_rotation_search=enable_rotation_search,
                enable_translation_search=enable_translation_search,
                bbox1=bbox1,
                bbox2=bbox2
            )
        except Exception as e:
            error_msg = f"比對過程中發生錯誤: {str(e)}"
            return False, 0.0, {'error': error_msg}, None, None

