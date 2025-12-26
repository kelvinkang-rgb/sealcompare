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
    
    def __init__(self, threshold: float = 0.95, 
                 similarity_ssim_weight: float = 0.5,
                 similarity_template_weight: float = 0.35,
                 pixel_similarity_weight: float = 0.1,
                 histogram_similarity_weight: float = 0.05):
        """
        初始化比對器
        
        Args:
            threshold: 相似度閾值，預設為 0.95（95%）
            similarity_ssim_weight: SSIM 權重，預設為 0.5 (50%)
            similarity_template_weight: Template Match 權重，預設為 0.35 (35%)
            pixel_similarity_weight: Pixel Similarity 權重，預設為 0.1 (10%)
            histogram_similarity_weight: Histogram Similarity 權重，預設為 0.05 (5%)
        """
        self.threshold = threshold
        self.similarity_ssim_weight = similarity_ssim_weight
        self.similarity_template_weight = similarity_template_weight
        self.pixel_similarity_weight = pixel_similarity_weight
        self.histogram_similarity_weight = histogram_similarity_weight
    
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
    
    def _align_image2_to_image1(
        self, 
        image1: np.ndarray, 
        image2: np.ndarray,
        rotation_range: float = 15.0,
        translation_range: int = 100
    ) -> Tuple[np.ndarray, float, Tuple[int, int], float, Dict[str, float], Dict[str, float]]:
        """
        將圖像2對齊到圖像1，使用相似度計算優化對齊參數
        
        Args:
            image1: 參考圖像（已去背景）
            image2: 待對齊圖像（已去背景）
            rotation_range: 旋轉角度搜索範圍（度）
            translation_range: 平移偏移搜索範圍（像素）
            
        Returns:
            (對齊後的圖像2, 最佳旋轉角度, (最佳x偏移, 最佳y偏移), 最佳相似度, 詳細指標, 階段時間字典)
        """
        import time
        alignment_timing = {}
        
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
        h1, w1 = img1_gray.shape
        
        # 階段1：快速旋轉搜索（分離旋轉和平移）
        stage1_start = time.time()
        # 使用縮小圖像進行快速評估
        scale_factor = 0.3  # 提高縮放比例以保持更多細節
        small_h = int(h2 * scale_factor)
        small_w = int(w2 * scale_factor)
        small_h1 = int(h1 * scale_factor)
        small_w1 = int(w1 * scale_factor)
        img1_small = cv2.resize(img1_gray, (small_w1, small_h1), interpolation=cv2.INTER_AREA)
        img2_small = cv2.resize(img2_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        # 粗搜索參數：使用更精細的步長提高初步精度
        angle_step_coarse = 3.0  # 從5度優化為3度，提高精度
        
        candidates = []  # 存儲候選 (angle, offset_x, offset_y, score)
        
        # 旋轉角度範圍：-rotation_range 到 +rotation_range
        for angle in np.arange(-rotation_range, rotation_range + angle_step_coarse, angle_step_coarse):
            # 限制在合理範圍內
            if angle < -80 or angle > 80:
                continue
            
            # 應用旋轉到縮小圖像
            center_small = (small_w // 2, small_h // 2)
            M_rot = cv2.getRotationMatrix2D(center_small, angle, 1.0)
            img2_rot_small = cv2.warpAffine(img2_small, M_rot, (small_w, small_h),
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=255)
            
            # 使用模板匹配快速估算平移（替代完整搜索）
            offset_x_est, offset_y_est, template_score = self._estimate_translation_template_match(
                img1_small, img2_rot_small, translation_range, scale_factor
            )
            
            # 如果模板匹配找到合理的平移，使用估算值；否則嘗試幾個候選偏移
            if template_score > 0.3:
                # 使用估算的平移
                candidates.append((angle, offset_x_est, offset_y_est, template_score))
            else:
                # 模板匹配失敗，嘗試幾個常見偏移（0, ±10, ±20像素）
                for dx_try in [0, -20, 20, -40, 40]:
                    for dy_try in [0, -20, 20, -40, 40]:
                        if abs(dx_try) > translation_range or abs(dy_try) > translation_range:
                            continue
                        dx_small = int(dx_try * scale_factor)
                        dy_small = int(dy_try * scale_factor)
                        M_trans = np.float32([[1, 0, dx_small], [0, 1, dy_small]])
                        img2_trans_small = cv2.warpAffine(img2_rot_small, M_trans, (small_w, small_h),
                                                         borderMode=cv2.BORDER_CONSTANT,
                                                         borderValue=255)
                        score = self._fast_rotation_match(img1_small, img2_trans_small)
                        candidates.append((angle, dx_try, dy_try, score))
        
        alignment_timing['stage1_coarse_search'] = time.time() - stage1_start
        
        # 按分數排序，動態調整候選數量
        if not candidates:
            # 如果沒有候選，使用默認值（不變換）
            best_angle = 0.0
            best_offset_x = 0
            best_offset_y = 0
            best_similarity = 0.0
            best_metrics = {}
        else:
            candidates.sort(key=lambda x: x[3], reverse=True)
            # 動態調整候選數量：根據分數差異決定
            # 如果最高分數很高（>0.9），減少候選數量；如果分數差異大，增加候選數量
            if len(candidates) > 1:
                score_diff = candidates[0][3] - candidates[-1][3]
                if candidates[0][3] > 0.9:
                    # 高分數時，只選擇前3個最佳候選
                    top_candidates = candidates[:3]
                elif score_diff > 0.3:
                    # 分數差異大時，選擇前5個
                    top_candidates = candidates[:5]
                else:
                    # 分數差異小時，選擇前7個以確保找到最佳匹配
                    top_candidates = candidates[:7]
            else:
                top_candidates = candidates[:5]
            
            # 階段2：完整相似度計算
            stage2_start = time.time()
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
                
                M_trans = np.float32([[1, 0, float(offset_x)], [0, 1, float(offset_y)]])
                img2_transformed = cv2.warpAffine(img2_rot, M_trans, (w2, h2),
                                                 borderMode=cv2.BORDER_CONSTANT,
                                                 borderValue=255)
                
                # 直接使用原始尺寸的圖像
                similarity = self._fast_rotation_match(img1_gray, img2_transformed)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_angle = angle
                    best_offset_x = offset_x
                    best_offset_y = offset_y
                    best_metrics = {}  # 不再使用 enhanced_metrics
            
            alignment_timing['stage2_full_evaluation'] = time.time() - stage2_start
        
        # 階段3：細搜索優化（在階段2的最佳值附近進行細搜索）
        if best_similarity > 0.0:
            stage3_start = time.time()
            # 直接使用階段2找到的最佳角度和平移值作為搜索中心
            # 在最佳值附近進行細搜索（縮小範圍以提高效率，保持精度）
            fine_angle_range = np.arange(-2.0, 2.1, 0.5)  # ±2度，每0.5度（從±3度優化）
            fine_offset_range = range(-5, 6, 1)  # ±5像素，每1像素（從±8像素優化）
            
            for angle_offset in fine_angle_range:
                angle = best_angle + angle_offset  # 以階段2的最佳角度為中心
                if angle < -80 or angle > 80:
                    continue
                
                # 應用旋轉
                center = (w2 // 2, h2 // 2)
                M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                img2_rot = cv2.warpAffine(img2_gray, M_rot, (w2, h2),
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=255)
                
                # 直接使用階段2的最佳平移值作為搜索中心
                for dx in fine_offset_range:
                    for dy in fine_offset_range:
                        offset_x = best_offset_x + dx  # 以階段2的最佳平移為中心
                        offset_y = best_offset_y + dy
                        
                        if abs(offset_x) > translation_range or abs(offset_y) > translation_range:
                            continue
                        
                        # 應用平移
                        M_trans = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
                        img2_transformed = cv2.warpAffine(img2_rot, M_trans, (w2, h2),
                                                          borderMode=cv2.BORDER_CONSTANT,
                                                          borderValue=255)
                        
                        # 計算相似度
                        similarity = self._fast_rotation_match(img1_gray, img2_transformed)
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_angle = angle
                            best_offset_x = offset_x
                            best_offset_y = offset_y
                            best_metrics = {}  # 不再使用 enhanced_metrics
            
            alignment_timing['stage3_fine_search'] = time.time() - stage3_start
        
        # 階段4：超細搜索優化（在階段3的最佳值附近進行最高精度搜索）
        if best_similarity > 0.0:
            stage4_start = time.time()
            # 直接使用階段3找到的最佳角度和平移值作為搜索中心
            # 在最佳值附近進行超細搜索（優化參數以平衡精度和效率）
            ultra_fine_angle_range = np.arange(-0.5, 0.51, 0.1)  # ±0.5度，每0.1度（保持）
            ultra_fine_offset_range = np.arange(-2.0, 2.1, 1.0)  # ±2像素，每1像素（從0.5像素優化）
            
            for angle_offset in ultra_fine_angle_range:
                angle = best_angle + angle_offset  # 以階段3的最佳角度為中心
                if angle < -80 or angle > 80:
                    continue
                
                # 應用旋轉
                center = (w2 // 2, h2 // 2)
                M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                img2_rot = cv2.warpAffine(img2_gray, M_rot, (w2, h2),
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=255)
                
                # 直接使用階段3的最佳平移值作為搜索中心
                for dx in ultra_fine_offset_range:
                    for dy in ultra_fine_offset_range:
                        offset_x = best_offset_x + dx  # 以階段3的最佳平移為中心
                        offset_y = best_offset_y + dy
                        
                        if abs(offset_x) > translation_range or abs(offset_y) > translation_range:
                            continue
                        
                        # 應用平移（使用浮點數精度）
                        M_trans = np.float32([[1, 0, float(offset_x)], [0, 1, float(offset_y)]])
                        img2_transformed = cv2.warpAffine(img2_rot, M_trans, (w2, h2),
                                                          borderMode=cv2.BORDER_CONSTANT,
                                                          borderValue=255)
                        
                        # 計算相似度
                        similarity = self._fast_rotation_match(img1_gray, img2_transformed)
                        
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_angle = angle
                            best_offset_x = offset_x
                            best_offset_y = offset_y
                            best_metrics = {}  # 不再使用 enhanced_metrics
            
            alignment_timing['stage4_ultra_fine_search'] = time.time() - stage4_start
        
        # 應用最佳變換到原始圖像（保持原始顏色）
        if len(image2.shape) == 3:
            img2_final = image2.copy()
        else:
            img2_final = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
        
        center = (w2 // 2, h2 // 2)
        M_rot = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        # 使用 INTER_LINEAR 插值以獲得更好的質量（默認是 INTER_LINEAR，但明確指定確保一致性）
        img2_rotated = cv2.warpAffine(img2_final, M_rot, (w2, h2),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
        
        # 確保使用浮點數精度
        M_trans = np.float32([[1, 0, float(best_offset_x)], [0, 1, float(best_offset_y)]])
        img2_aligned = cv2.warpAffine(img2_rotated, M_trans, (w2, h2),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
        
        return img2_aligned, best_angle, (best_offset_x, best_offset_y), best_similarity, best_metrics, alignment_timing
    
    def _estimate_translation_template_match(
        self, 
        img1: np.ndarray, 
        img2: np.ndarray, 
        translation_range: int,
        scale_factor: float
    ) -> Tuple[int, int, float]:
        """
        使用模板匹配快速估算平移偏移
        
        Args:
            img1: 參考圖像（已縮放）
            img2: 待匹配圖像（已旋轉，未平移）
            translation_range: 平移搜索範圍（原始尺寸）
            scale_factor: 圖像縮放比例
            
        Returns:
            (估算的x偏移, 估算的y偏移, 匹配分數)
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 確保模板不大於搜索圖像
        if w2 > w1 or h2 > h1:
            # 尺寸不兼容，返回默認值
            return 0, 0, 0.0
        
        # 模板匹配（img2作為模板在img1中搜索）
        result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        # max_loc 是模板（img2）左上角在 img1 中的位置
        # 計算兩個圖像中心的偏移
        center_x1 = w1 // 2
        center_y1 = h1 // 2
        center_x2 = w2 // 2
        center_y2 = h2 // 2
        
        # 匹配位置的中心點（相對於 img1）
        match_center_x = max_loc[0] + center_x2
        match_center_y = max_loc[1] + center_y2
        
        # 計算偏移：img2 的中心需要移動多少才能對齊到 img1 的中心
        # 如果 match_center_x < center_x1，說明 img2 在左邊，需要向右移動（正偏移）
        offset_x_scaled = center_x1 - match_center_x
        offset_y_scaled = center_y1 - match_center_y
        
        # 轉換回原始尺寸的偏移
        offset_x = int(offset_x_scaled / scale_factor)
        offset_y = int(offset_y_scaled / scale_factor)
        
        # 限制在搜索範圍內
        offset_x = max(-translation_range, min(translation_range, offset_x))
        offset_y = max(-translation_range, min(translation_range, offset_y))
        
        return offset_x, offset_y, float(max_val)
    
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
        # 方法1：邊緣檢測 + 模板匹配（最快且對旋轉敏感）
        edges1 = cv2.Canny(img1, 50, 150)
        edges2 = cv2.Canny(img2, 50, 150)
        
        edge_match = 0.0
        if np.sum(edges1) > 0 and np.sum(edges2) > 0:
            # 使用模板匹配（確保模板不大於搜索圖像）
            h1, w1 = edges1.shape
            h2, w2 = edges2.shape
            if h2 <= h1 and w2 <= w1:
                # edges2 可以作為模板在 edges1 中搜索
                result = cv2.matchTemplate(edges1, edges2, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                edge_match = float(max_val)
            elif h1 <= h2 and w1 <= w2:
                # edges1 可以作為模板在 edges2 中搜索
                result = cv2.matchTemplate(edges2, edges1, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                edge_match = float(max_val)
            else:
                # 尺寸不兼容，跳過模板匹配
                edge_match = 0.0
        elif np.sum(edges1) == 0 and np.sum(edges2) == 0:
            # 如果都沒有邊緣，可能是空白圖像
            edge_match = 0.5
        
        # 方法2：簡化版SSIM（結構相似性）- 需要相同尺寸，裁剪到較小尺寸
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        if h1 == h2 and w1 == w2:
            # 尺寸相同，直接計算
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
        else:
            # 尺寸不同，裁剪到較小尺寸
            min_h = min(h1, h2)
            min_w = min(w1, w2)
            img1_crop = img1[:min_h, :min_w]
            img2_crop = img2[:min_h, :min_w]
            
            img1_float = img1_crop.astype(np.float64)
            img2_float = img2_crop.astype(np.float64)
            
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
        
        # 方法3：像素相似度（快速，但對旋轉不敏感）- 需要相同尺寸，裁剪到較小尺寸
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        if h1 == h2 and w1 == w2:
            # 尺寸相同，直接計算
            mask1 = img1 < 250  # 非背景像素
            mask2 = img2 < 250
            mask_combined = mask1 & mask2
            
            pixel_similarity = 0.0
            if np.sum(mask_combined) > 0:
                diff = cv2.absdiff(img1, img2)
                pixel_similarity = 1.0 - (np.sum(diff[mask_combined] > 10) / np.sum(mask_combined))
        else:
            # 尺寸不同，裁剪到較小尺寸
            min_h = min(h1, h2)
            min_w = min(w1, w2)
            img1_crop = img1[:min_h, :min_w]
            img2_crop = img2[:min_h, :min_w]
            
            mask1 = img1_crop < 250  # 非背景像素
            mask2 = img2_crop < 250
            mask_combined = mask1 & mask2
            
            pixel_similarity = 0.0
            if np.sum(mask_combined) > 0:
                diff = cv2.absdiff(img1_crop, img2_crop)
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
        
        # 直接使用原始尺寸的圖像（移除尺寸調整以避免圖像變形）
        img1_final = img1_processed
        img2_final = img2_processed
        
        # 檢查圖像尺寸
        h1, w1 = img1_final.shape[:2]
        h2, w2 = img2_final.shape[:2]
        print(f"圖像尺寸: img1=({h1}, {w1}), img2=({h2}, {w2})")
        
        # 直接計算相似度指標（優化：只計算使用的指標，添加錯誤處理）
        try:
            similarity_ssim = self._calculate_ssim(img1_final, img2_final)
            print(f"SSIM 相似度: {similarity_ssim:.4f}")
        except Exception as e:
            print(f"SSIM 計算失敗: {e}")
            similarity_ssim = 0.0
        
        try:
            similarity_template = self._template_match(img1_final, img2_final)
            print(f"模板匹配相似度: {similarity_template:.4f}")
        except Exception as e:
            print(f"模板匹配計算失敗: {e}")
            similarity_template = 0.0
        
        try:
            pixel_diff = self._pixel_difference(img1_final, img2_final)
            pixel_similarity = 1.0 - pixel_diff
            print(f"像素相似度: {pixel_similarity:.4f}")
        except Exception as e:
            print(f"像素相似度計算失敗: {e}")
            pixel_similarity = 0.0
        
        try:
            # 計算直方圖相似度
            hist1 = cv2.calcHist([img1_final], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2_final], [0], None, [256], [0, 256])
            histogram_similarity = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))
            print(f"直方圖相似度: {histogram_similarity:.4f}")
        except Exception as e:
            print(f"直方圖相似度計算失敗: {e}")
            histogram_similarity = 0.0
        
        # 使用簡單的加權組合計算最終相似度（優化權重，增強抗背景噪訊能力）
        # 提高結構特徵演算法權重（SSIM, Template Match），降低像素級演算法權重（Pixel, Histogram）
        similarity = (
            similarity_ssim * self.similarity_ssim_weight +
            similarity_template * self.similarity_template_weight +
            pixel_similarity * self.pixel_similarity_weight +
            histogram_similarity * self.histogram_similarity_weight
        )
        
        print(f"最終相似度: {similarity:.4f} (SSIM={similarity_ssim:.4f}, Template={similarity_template:.4f}, Pixel={pixel_similarity:.4f}, Hist={histogram_similarity:.4f})")
        
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
        # 檢查並統一圖像尺寸
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if h1 != h2 or w1 != w2:
            # 尺寸不同，調整到相同尺寸（使用較小尺寸以保持精度）
            target_h = min(h1, h2)
            target_w = min(w1, w2)
            img1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
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
        # 檢查並統一圖像尺寸
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if h1 != h2 or w1 != w2:
            # 尺寸不同，調整到相同尺寸（使用較小尺寸以保持精度）
            target_h = min(h1, h2)
            target_w = min(w1, w2)
            img1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        diff = cv2.absdiff(img1, img2)
        diff_pixels = np.count_nonzero(diff)
        total_pixels = img1.size
        
        return diff_pixels / total_pixels if total_pixels > 0 else 0.0
    
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

