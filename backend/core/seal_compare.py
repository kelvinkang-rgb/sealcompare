"""
印鑑比對模組
用於比對兩個印章圖像是否完全一致
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import hashlib


class SealComparator:
    """印鑑比對器"""
    
    def __init__(self, threshold: float = 0.83, 
                 similarity_ssim_weight: float = 0.5,
                 similarity_template_weight: float = 0.35,
                 pixel_similarity_weight: float = 0.1,
                 histogram_similarity_weight: float = 0.05):
        """
        初始化比對器
        
        Args:
            threshold: 相似度閾值，預設為 0.83（83%）
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
    
    def _auto_detect_bounds_and_remove_background(self, image: np.ndarray, return_timing: bool = False) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        自動偵測圖像外框並移除背景顏色
        
        使用演算法自動偵測圖像的實際內容邊界，並移除背景（通常是白紙的顏色）
        
        Args:
            image: 輸入圖像（可以是彩色或灰度）
            return_timing: 是否返回時間詳情（如果為False，返回空字典）
            
        Returns:
            (裁切並移除背景後的圖像, 時間詳情字典)
            注意：如果 return_timing=False，時間詳情字典為空
        """
        import time
        timing = {}
        
        if image is None or image.size == 0:
            return image, timing
        
        # 步驟1：轉換為灰度圖
        step1_start = time.time()
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        timing['step1_convert_to_gray'] = time.time() - step1_start

        # 陰影/摺痕抑制：低頻光照校正（讓背景更均勻，避免陰影被 OTSU 當作前景）
        shade_start = time.time()
        corrected_gray = gray
        try:
            # kernel 尺寸隨影像大小動態調整（取較小邊的 1/12，並強制為奇數且至少 31）
            ksize = max(31, int(min(gray.shape) / 12) | 1)
            kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_bg)
            corrected_gray = cv2.addWeighted(gray, 1.0, background, -1.0, 255.0)
        except Exception:
            corrected_gray = gray
        timing['step1b_shading_correction'] = time.time() - shade_start
        
        h, w = corrected_gray.shape
        
        # 步驟2：檢測背景顏色（分析圖像邊緣）
        step2_start = time.time()
        edge_width = max(5, min(h, w) // 20)
        edge_pixels = []
        edge_pixels.extend(corrected_gray[0:edge_width, :].flatten())
        edge_pixels.extend(corrected_gray[h-edge_width:h, :].flatten())
        edge_pixels.extend(corrected_gray[:, 0:edge_width].flatten())
        edge_pixels.extend(corrected_gray[:, w-edge_width:w].flatten())
        
        # 計算背景顏色的中位數和標準差
        edge_pixels_array = np.array(edge_pixels)
        bg_color = np.median(edge_pixels_array)
        bg_std = np.std(edge_pixels_array)
        
        # 背景閾值：背景顏色 ± 2倍標準差（通常背景是白色，值接近255）
        bg_threshold_low = max(200, int(bg_color - 2 * bg_std))
        bg_threshold_high = 255
        timing['step2_detect_bg_color'] = time.time() - step2_start
        
        # 步驟3：OTSU 二值化
        step3_start = time.time()
        _, binary_otsu = cv2.threshold(corrected_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        timing['step3_otsu_threshold'] = time.time() - step3_start

        # 線條型雜訊抑制：移除細長且面積小的連通元件（常見於印刷黑線/邊框/掃描雜線）
        line_start = time.time()
        try:
            fg = (binary_otsu > 0).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
            if n > 2:
                areas = stats[1:, cv2.CC_STAT_AREA]
                max_area = int(areas.max()) if areas.size else 0
                cleaned = fg.copy()
                for i in range(1, n):
                    w_i = int(stats[i, cv2.CC_STAT_WIDTH])
                    h_i = int(stats[i, cv2.CC_STAT_HEIGHT])
                    area = int(stats[i, cv2.CC_STAT_AREA])
                    if w_i <= 0 or h_i <= 0:
                        continue
                    aspect = max(w_i, h_i) / max(1, min(w_i, h_i))
                    # 過濾條件：細長（高長寬比）+ 面積相對小
                    if max_area > 0 and aspect >= 8.0 and area <= max(120, int(max_area * 0.02)):
                        cleaned[labels == i] = 0
                binary_otsu = (cleaned * 255).astype(np.uint8)
        except Exception:
            pass
        timing['step3b_remove_thin_lines'] = time.time() - line_start
        
        # 步驟4：結合背景顏色檢測和二值化結果
        step4_start = time.time()
        # 創建背景遮罩：接近背景顏色的像素
        bg_mask_color = (corrected_gray >= bg_threshold_low) & (corrected_gray <= bg_threshold_high)
        
        # 結合 OTSU 結果：OTSU 的前景區域（值為255）應該保留
        # OTSU 的前景是255，背景是0，所以背景遮罩應該是 OTSU 背景（0）且顏色接近背景
        bg_mask_otsu = (binary_otsu == 0)  # OTSU 的背景區域
        
        # 結合兩個條件：既是背景顏色，又是 OTSU 識別的背景
        bg_mask = bg_mask_color & bg_mask_otsu
        timing['step4_combine_masks'] = time.time() - step4_start
        
        # 步驟6：輪廓檢測找到實際內容的邊界框
        step6_start = time.time()
        # 創建前景遮罩（非背景區域）
        foreground_mask = ~bg_mask
        
        # 找到輪廓
        contours, _ = cv2.findContours(foreground_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        timing['step6_contour_detection'] = time.time() - step6_start
        
        if len(contours) == 0:
            # 如果沒有找到輪廓（常見於「已裁切的小印鑑圖」），不要直接返回；
            # 改用紅章分割/簡易背景移除作為 fallback，避免黑線/摺痕殘留。
            if len(image.shape) == 3:
                try:
                    fallback_start = time.time()
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    lower1 = np.array([0, 50, 40], dtype=np.uint8)
                    upper1 = np.array([10, 255, 255], dtype=np.uint8)
                    lower2 = np.array([170, 50, 40], dtype=np.uint8)
                    upper2 = np.array([180, 255, 255], dtype=np.uint8)
                    mask1 = cv2.inRange(hsv, lower1, upper1)
                    mask2 = cv2.inRange(hsv, lower2, upper2)
                    red_mask_hsv = cv2.bitwise_or(mask1, mask2)
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    a = lab[:, :, 1]
                    red_mask_lab = (a > 150).astype(np.uint8) * 255
                    red_mask = cv2.bitwise_or(red_mask_hsv, red_mask_lab)
                    if float(np.mean(red_mask > 0)) >= 0.008:
                        result = image.copy()
                        result[red_mask == 0] = [255, 255, 255]
                        timing['step6_fallback_red_seal_segmentation'] = time.time() - fallback_start
                        if return_timing and timing:
                            timing['remove_background_total'] = sum(v for k, v in timing.items() if k != 'remove_background_total')
                        return result, timing
                except Exception:
                    pass

            return image, timing
        
        # 步驟7：找到最大輪廓並計算邊界框
        step7_start = time.time()
        # 找到最大的輪廓（假設是印章）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 檢查輪廓面積是否足夠大（至少佔圖像的1%）
        contour_area = cv2.contourArea(largest_contour)
        min_area = (h * w) * 0.01
        if contour_area < min_area:
            # 輪廓太小（可能是已裁切印鑑或噪點），同樣嘗試 fallback 去背景
            if len(image.shape) == 3:
                try:
                    fallback_start = time.time()
                    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    lower1 = np.array([0, 50, 40], dtype=np.uint8)
                    upper1 = np.array([10, 255, 255], dtype=np.uint8)
                    lower2 = np.array([170, 50, 40], dtype=np.uint8)
                    upper2 = np.array([180, 255, 255], dtype=np.uint8)
                    mask1 = cv2.inRange(hsv, lower1, upper1)
                    mask2 = cv2.inRange(hsv, lower2, upper2)
                    red_mask_hsv = cv2.bitwise_or(mask1, mask2)
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    a = lab[:, :, 1]
                    red_mask_lab = (a > 150).astype(np.uint8) * 255
                    red_mask = cv2.bitwise_or(red_mask_hsv, red_mask_lab)
                    if float(np.mean(red_mask > 0)) >= 0.008:
                        result = image.copy()
                        result[red_mask == 0] = [255, 255, 255]
                        timing['step7_fallback_red_seal_segmentation'] = time.time() - fallback_start
                        if return_timing and timing:
                            timing['remove_background_total'] = sum(v for k, v in timing.items() if k != 'remove_background_total')
                        return result, timing
                except Exception:
                    pass

            return image, timing
        
        # 計算邊界框
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 確保邊界框有效
        if w <= 0 or h <= 0:
            return image, timing
        
        # 添加足夠的邊距（10-15%），確保印鑑內容不被切掉
        margin_ratio = 0.12  # 12% 邊距
        margin = max(10, int(min(h, w) * margin_ratio))  # 至少10像素，或12%的較小邊
        x = max(0, x - margin)
        y = max(0, y - margin)
        # 確保不會超出圖像邊界
        w = min(w + 2 * margin, corrected_gray.shape[1] - x)
        h = min(h + 2 * margin, corrected_gray.shape[0] - y)
        
        # 再次確保邊界框有效
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > corrected_gray.shape[1] or y + h > corrected_gray.shape[0]:
            return image, timing
        timing['step7_calculate_bbox'] = time.time() - step7_start
        
        # 步驟8：裁切圖像
        step8_start = time.time()
        # 裁切圖像到邊界框
        if len(image.shape) == 3:
            cropped = image[y:y+h, x:x+w].copy()
        else:
            cropped = gray[y:y+h, x:x+w].copy()
        timing['step8_crop_image'] = time.time() - step8_start
        
        # 步驟9：在裁切後的圖像上重新檢測背景並移除
        step9_start = time.time()
        # 移除背景：將背景區域設為白色（255）
        if len(cropped.shape) == 3:
            # 彩色圖像：轉換為灰度來檢測背景
            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        else:
            cropped_gray = cropped.copy()
        
        # 在裁切後的圖像上重新檢測背景並移除
        if cropped.size == 0:
            return image, timing

        # === 紅章優先去背景（對「黑線/摺痕陰影」特別有效）===
        # 若裁切後圖像存在足夠的紅色像素，直接以紅色前景 mask 作為印鑑，
        # 將非紅色區域全部視為背景並設為白色，避免黑線/陰影被保留。
        if len(cropped.shape) == 3:
            try:
                hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
                # HSV 紅色兩段（hue 0-10, 170-180）
                lower1 = np.array([0, 50, 40], dtype=np.uint8)
                upper1 = np.array([10, 255, 255], dtype=np.uint8)
                lower2 = np.array([170, 50, 40], dtype=np.uint8)
                upper2 = np.array([180, 255, 255], dtype=np.uint8)
                mask1 = cv2.inRange(hsv, lower1, upper1)
                mask2 = cv2.inRange(hsv, lower2, upper2)
                red_mask_hsv = cv2.bitwise_or(mask1, mask2)

                # Lab 的 a-channel：紅色偏向 a > 128（越紅越高）
                lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
                a = lab[:, :, 1]
                red_mask_lab = (a > 150).astype(np.uint8) * 255

                red_mask = cv2.bitwise_or(red_mask_hsv, red_mask_lab)

                # 過濾小型紅噪點（保留主要連通元件）
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((red_mask > 0).astype(np.uint8), connectivity=8)
                if num_labels > 1:
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    max_area = int(areas.max()) if areas.size else 0
                    keep = np.zeros_like(red_mask, dtype=np.uint8)
                    # 保留面積接近最大者（避免保留散落小紅點）
                    for i in range(1, num_labels):
                        area = int(stats[i, cv2.CC_STAT_AREA])
                        if max_area > 0 and area >= max(80, int(max_area * 0.02)):
                            keep[labels == i] = 255
                    red_mask = keep

                red_ratio = float(np.mean(red_mask > 0))
                # 紅色比例太低則視為非紅章，走 fallback
                if red_ratio >= 0.008:
                    result = cropped.copy()
                    result[red_mask == 0] = [255, 255, 255]
                    timing['step9_remove_bg_final'] = time.time() - step9_start

                    # 計算總時間（僅在需要時計算）
                    if return_timing and timing:
                        timing['remove_background_total'] = sum(v for k, v in timing.items() if k != 'remove_background_total')

                    return result, timing
            except Exception:
                # 若紅章分割失敗，退回原本灰階背景移除邏輯
                pass
        
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
        timing['step9_remove_bg_final'] = time.time() - step9_start
        
        # 計算總時間（僅在需要時計算）
        if return_timing and timing:
            # 排除總時間本身，避免重複計算
            timing['remove_background_total'] = sum(v for k, v in timing.items() if k != 'remove_background_total')
        
        return result, timing
    
    def _align_image2_to_image1(
        self, 
        image1: np.ndarray, 
        image2: np.ndarray,
        rotation_range: float = 15.0,
        translation_range: int = 100
    ) -> Tuple[np.ndarray, float, Tuple[int, int], float, Dict[str, float], Dict[str, float]]:
        """
        將圖像2對齊到圖像1，使用分離式搜索流程優化對齊參數
        
        新流程：
        1. 平移粗調：使用模板匹配快速估算平移
        2. 旋轉粗調：固定平移值，只搜索旋轉角度
        3. 平移細調：在最佳旋轉角度下精細調整平移
        4. 旋轉細調與平移細調：交替優化（先旋轉再平移，重複幾次）
        
        Args:
            image1: 參考圖像（已去背景）
            image2: 待對齊圖像（已去背景）
            rotation_range: 旋轉角度搜索範圍（度）
            translation_range: 平移偏移搜索範圍（像素，僅作為參考，實際使用動態計算的範圍）
            
        Returns:
            (對齊後的圖像2, 最佳旋轉角度, (最佳x偏移, 最佳y偏移), 最佳相似度, 詳細指標, 階段時間字典)
            
        注意：
            - 為了達到極高精度要求，平移範圍會根據圖像尺寸動態計算為 max(w1, h1, w2, h2)
            - 這確保能覆蓋整個圖像範圍內的所有可能偏移
        """
        import time
        alignment_timing = {}
        best_metrics = {}

        # === 可觀測性：預先填入（即使未觸發也有 key）===
        best_metrics['angle_sign_check_triggered'] = False
        best_metrics['angle_sign_flipped'] = False
        best_metrics['overlap_before_sign_check'] = None
        best_metrics['overlap_after_sign_check'] = None
        
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
        
        # 動態計算平移範圍：使用較大圖像的尺寸作為範圍，確保能覆蓋整個圖像可能的偏移
        # 為了極高精度要求（比對pixel差異數量），使用最大尺寸作為平移範圍
        # 這確保不會因為平移範圍不足而錯過最佳匹配位置
        max_dimension = max(w1, h1, w2, h2)
        dynamic_translation_range = max_dimension  # 使用整個圖像範圍
        
        # 階段1+2（合併）：joint-grid 粗搜尋（角度×平移），並以多尺度模板匹配強化平移
        stage12_start = time.time()
        print(f"對齊開始：圖像尺寸 img1=({h1}, {w1}), img2=({h2}, {w2}), 平移範圍={dynamic_translation_range}")
        try:
            best_angle, best_offset_x, best_offset_y, coarse_similarity, stage12_timing = self._joint_coarse_search_multiscale(
                img1_gray,
                img2_gray,
                rotation_range=rotation_range,
                translation_range=dynamic_translation_range
            )
            # 記錄 coarse 模式，方便除錯
            best_metrics['coarse_search_mode'] = "joint_grid"
            best_metrics['coarse_similarity'] = float(coarse_similarity)
            # 合併細節時間
            if stage12_timing:
                alignment_timing.update(stage12_timing)
            print(
                f"階段1+2(joint)完成：角度={best_angle:.2f}度, 偏移=({best_offset_x}, {best_offset_y}), "
                f"相似度={coarse_similarity:.4f}"
            )
        except Exception as e:
            print(f"警告：joint粗搜尋失敗，使用默認值: {str(e)}")
            best_angle = 0.0
            best_offset_x = 0
            best_offset_y = 0
            coarse_similarity = 0.0
            best_metrics['coarse_search_mode'] = "joint_grid_failed"
            best_metrics['coarse_error'] = str(e)

        stage12_total = time.time() - stage12_start
        # timing 相容：保留 stage1/stage2 key，並新增 stage12_joint_coarse_total
        alignment_timing['stage12_joint_coarse_total'] = alignment_timing.get('stage12_joint_coarse_total', stage12_total)
        alignment_timing['stage1_translation_coarse'] = stage12_total
        alignment_timing['stage2_rotation_coarse'] = 0.0

        # === 可觀測性：記錄 stage12 後的 overlap 與 offset（用縮小圖快速估計）===
        try:
            best_metrics['offset_after_stage12'] = {'x': int(best_offset_x), 'y': int(best_offset_y), 'angle': float(best_angle)}
            best_metrics['overlap_after_stage12'] = float(
                self._fast_overlap_ratio(img1_gray, img2_gray, best_angle, best_offset_x, best_offset_y, scale=0.3)
            )
        except Exception as e:
            best_metrics['overlap_after_stage12_error'] = str(e)

        # === 低 overlap multi-start：用 stage12 的 top-K 候選重新以 overlap 評估，避免落錯峰 ===
        try:
            best_metrics['multistart_triggered'] = False
            overlap_stage12 = best_metrics.get('overlap_after_stage12')
            candidates = alignment_timing.get('stage12_candidates') if isinstance(alignment_timing, dict) else None
            if overlap_stage12 is not None and overlap_stage12 < 0.6 and isinstance(candidates, list) and len(candidates) > 1:
                best_metrics['multistart_triggered'] = True
                scored = []
                for c in candidates:
                    try:
                        a = float(c.get('angle'))
                        dx = int(c.get('dx'))
                        dy = int(c.get('dy'))
                        ov = float(self._fast_overlap_ratio(img1_gray, img2_gray, a, dx, dy, scale=0.3))
                        scored.append((ov, a, dx, dy))
                    except Exception:
                        continue
                if scored:
                    scored.sort(key=lambda x: x[0], reverse=True)
                    best_ov, best_a, best_dx, best_dy = scored[0]
                    best_metrics['multistart_best_overlap'] = float(best_ov)
                    best_metrics['multistart_best_candidate'] = {'angle': float(best_a), 'dx': int(best_dx), 'dy': int(best_dy)}
                    # 若候選 overlap 明顯優於原本結果，採用它作為後續精修起點
                    if best_ov > float(overlap_stage12) + 0.05:
                        best_angle = best_a
                        best_offset_x = best_dx
                        best_offset_y = best_dy
                        best_metrics['multistart_adopted'] = True
                    else:
                        best_metrics['multistart_adopted'] = False
        except Exception as e:
            best_metrics['multistart_error'] = str(e)

        # ===== 平移救援（優先救大錯峰）=====
        # 用縮小圖 overlap_ratio 判斷是否落錯峰；若低於門檻，固定角度做更大半徑平移搜尋（±120，8→3→1）
        rescue_start = time.time()
        try:
            overlap_before = self._fast_overlap_ratio(img1_gray, img2_gray, best_angle, best_offset_x, best_offset_y, scale=0.3)
            best_metrics['rescue_triggered'] = False
            best_metrics['rescue_overlap_before'] = float(overlap_before)

            if overlap_before < 0.6:
                best_metrics['rescue_triggered'] = True
                rx, ry, ob, oa, rescue_timing = self._translation_rescue_search(
                    img1_gray,
                    img2_gray,
                    angle=best_angle,
                    initial_offset_x=best_offset_x,
                    initial_offset_y=best_offset_y,
                    overlap_threshold=0.6,
                    radius=120,
                    steps=(8, 3, 1),
                    scale=0.3
                )
                best_offset_x = rx
                best_offset_y = ry
                best_metrics['rescue_overlap_before'] = float(ob)
                best_metrics['rescue_overlap_after'] = float(oa)
                if rescue_timing:
                    alignment_timing.update(rescue_timing)
        except Exception as e:
            best_metrics['rescue_triggered'] = None
            best_metrics['rescue_error'] = str(e)

        alignment_timing['stage3_translation_rescue_total'] = alignment_timing.get(
            'stage3_translation_rescue_total',
            time.time() - rescue_start
        )
        # ===== 平移救援結束 =====

        # ===== 角度正負判別（sign-disambiguation）：避免 +θ/-θ 選錯 =====
        try:
            ov_now = float(self._fast_overlap_ratio(img1_gray, img2_gray, best_angle, best_offset_x, best_offset_y, scale=0.3))
            # 只在低 overlap 時啟動（避免不必要計算）
            if ov_now < 0.6 and abs(float(best_angle)) >= 0.5:
                a2, dx2, dy2, sign_metrics = self._angle_sign_disambiguation(
                    img1_gray,
                    img2_gray,
                    angle=float(best_angle),
                    offset_x=int(best_offset_x),
                    offset_y=int(best_offset_y),
                    scale=0.3,
                    improve_threshold=0.05,
                    rescue_radius=60,
                    rescue_steps=(6, 2, 1)
                )
                best_metrics.update(sign_metrics)
                if sign_metrics.get('angle_sign_flipped'):
                    best_angle = float(a2)
                    best_offset_x = int(dx2)
                    best_offset_y = int(dy2)
        except Exception as e:
            best_metrics['angle_sign_check_error_outer'] = str(e)
        # ===== 角度正負判別結束 =====
        
        # 階段3：平移細調（在最佳旋轉角度下精細調整平移，1像素精度）
        stage3_start = time.time()
        try:
            refined_offset_x, refined_offset_y, fine_similarity = self._fine_translation_search(
                img1_gray, img2_gray, best_angle, best_offset_x, best_offset_y, search_range=8
            )
            best_offset_x = refined_offset_x
            best_offset_y = refined_offset_y
            best_similarity = fine_similarity
            print(f"階段3平移細調完成：偏移=({best_offset_x}, {best_offset_y}), 相似度={best_similarity:.4f}")
        except Exception as e:
            print(f"警告：平移細調失敗，使用粗調結果: {str(e)}")
            # 評估當前最佳值
            center = (w2 // 2, h2 // 2)
            M_rot = cv2.getRotationMatrix2D(center, best_angle, 1.0)
            img2_rot = cv2.warpAffine(img2_gray, M_rot, (w2, h2),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=255)
            M_trans = np.float32([[1, 0, float(best_offset_x)], [0, 1, float(best_offset_y)]])
            img2_transformed = cv2.warpAffine(img2_rot, M_trans, (w2, h2),
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=255)
            best_similarity = self._fast_rotation_match(img1_gray, img2_transformed)
        
        alignment_timing['stage3_translation_fine'] = time.time() - stage3_start

        # === 可觀測性：記錄 stage3 後的 overlap 與 offset ===
        try:
            best_metrics['offset_after_stage3'] = {'x': int(best_offset_x), 'y': int(best_offset_y), 'angle': float(best_angle)}
            best_metrics['overlap_after_stage3'] = float(
                self._fast_overlap_ratio(img1_gray, img2_gray, best_angle, best_offset_x, best_offset_y, scale=0.3)
            )
        except Exception as e:
            best_metrics['overlap_after_stage3_error'] = str(e)
        
        # 階段4：旋轉細調與平移細調（交替優化，0.2度旋轉精度，1像素平移精度）
        # 優化：根據階段3的相似度動態調整迭代次數
        stage4_start = time.time()
        try:
            # 如果階段3的相似度已經很高（>0.98），減少迭代次數
            if best_similarity > 0.98:
                iterations = 1
                print(f"階段4：相似度 {best_similarity:.4f} 已很高，使用1次迭代")
            else:
                iterations = 2
                print(f"階段4：相似度 {best_similarity:.4f}，使用2次迭代")
            
            final_angle, final_offset_x, final_offset_y, final_similarity, stage4_timing = self._alternating_fine_tuning(
                img1_gray, img2_gray, best_angle, best_offset_x, best_offset_y,
                rotation_range=2.0, translation_range=3, iterations=iterations
            )
            
            # 如果交替優化找到更好的結果，使用它
            if final_similarity > best_similarity:
                improvement = final_similarity - best_similarity
                best_angle = final_angle
                best_offset_x = final_offset_x
                best_offset_y = final_offset_y
                best_similarity = final_similarity
                print(f"階段4交替優化完成：角度={best_angle:.2f}度, 偏移=({best_offset_x}, {best_offset_y}), "
                      f"相似度={best_similarity:.4f}, 改進={improvement:.4f}")
            else:
                print(f"階段4交替優化：未發現更好的解，保持階段3結果")
            
            # 合併階段4的時間詳情
            alignment_timing.update(stage4_timing)
        except Exception as e:
            print(f"警告：交替優化失敗，使用階段3結果: {str(e)}")
            alignment_timing['stage4_rotation_fine'] = 0.0
            alignment_timing['stage4_translation_fine'] = 0.0
            alignment_timing['stage4_total'] = 0.0
        
        alignment_timing['stage4_total'] = alignment_timing.get('stage4_total', time.time() - stage4_start)

        # === 可觀測性：記錄 stage4 後的 overlap 與 offset ===
        try:
            best_metrics['offset_after_stage4'] = {'x': int(best_offset_x), 'y': int(best_offset_y), 'angle': float(best_angle)}
            best_metrics['overlap_after_stage4'] = float(
                self._fast_overlap_ratio(img1_gray, img2_gray, best_angle, best_offset_x, best_offset_y, scale=0.3)
            )
        except Exception as e:
            best_metrics['overlap_after_stage4_error'] = str(e)

        # === 二次救援（post-fine guardrail）：防止精修後仍落錯峰/漂走 ===
        post_rescue_start = time.time()
        try:
            best_metrics['post_fine_rescue_triggered'] = False
            overlap12 = best_metrics.get('overlap_after_stage12')
            overlap4 = best_metrics.get('overlap_after_stage4')
            # 觸發條件：overlap 太低，或相比 stage12 明顯下降
            trigger = False
            if overlap4 is not None and overlap4 < 0.6:
                trigger = True
            if overlap12 is not None and overlap4 is not None and overlap4 < float(overlap12) - 0.15:
                trigger = True

            if trigger:
                best_metrics['post_fine_rescue_triggered'] = True

                # 先嘗試角度正負判別，避免「角度錯了卻一直只救平移」
                try:
                    a2, dx2, dy2, sign2 = self._angle_sign_disambiguation(
                        img1_gray,
                        img2_gray,
                        angle=float(best_angle),
                        offset_x=int(best_offset_x),
                        offset_y=int(best_offset_y),
                        scale=0.3,
                        improve_threshold=0.05,
                        rescue_radius=60,
                        rescue_steps=(6, 2, 1)
                    )
                    # 只要這裡 flip，後面的平移救援就以新角度為準（或甚至不需要）
                    best_metrics['post_fine_angle_sign_metrics'] = sign2
                    if sign2.get('angle_sign_flipped'):
                        best_angle = float(a2)
                        best_offset_x = int(dx2)
                        best_offset_y = int(dy2)
                except Exception as e_sign:
                    best_metrics['post_fine_angle_sign_error'] = str(e_sign)

                rx, ry, ob, oa, rescue_timing = self._translation_rescue_search(
                    img1_gray,
                    img2_gray,
                    angle=best_angle,
                    initial_offset_x=best_offset_x,
                    initial_offset_y=best_offset_y,
                    overlap_threshold=0.65,
                    radius=80,
                    steps=(6, 2, 1),
                    scale=0.3
                )
                best_metrics['post_fine_rescue_overlap_before'] = float(ob)
                best_metrics['post_fine_rescue_overlap_after'] = float(oa)
                if rescue_timing:
                    # 避免覆蓋 stage3 的 key，這裡用獨立命名
                    alignment_timing['post_fine_rescue_total'] = rescue_timing.get('stage3_translation_rescue_total', 0.0)
                    alignment_timing['post_fine_rescue_evals'] = rescue_timing.get('stage3_translation_rescue_evals', 0.0)

                best_offset_x = rx
                best_offset_y = ry

                # 再跑一次小範圍 stage3 fine，確保收斂
                try:
                    refined_x2, refined_y2, sim2 = self._fine_translation_search(
                        img1_gray, img2_gray, best_angle, best_offset_x, best_offset_y, search_range=10
                    )
                    best_offset_x = refined_x2
                    best_offset_y = refined_y2
                    best_similarity = max(best_similarity, sim2)
                    best_metrics['post_fine_rescue_offset'] = {'x': int(best_offset_x), 'y': int(best_offset_y)}
                    best_metrics['post_fine_rescue_overlap_final'] = float(
                        self._fast_overlap_ratio(img1_gray, img2_gray, best_angle, best_offset_x, best_offset_y, scale=0.3)
                    )
                except Exception as e2:
                    best_metrics['post_fine_rescue_fine_error'] = str(e2)
        except Exception as e:
            best_metrics['post_fine_rescue_error'] = str(e)
        alignment_timing['post_fine_rescue_guardrail_total'] = time.time() - post_rescue_start
        
        # 階段5：全局驗證（確保找到全局最優解）
        # 優化：根據當前相似度自適應調整搜索範圍
        stage5_start = time.time()
        try:
            # 自適應搜索範圍調整
            if best_similarity > 0.99:
                # 相似度已經很高，跳過全局驗證或使用極小範圍
                print(f"階段5：相似度已達 {best_similarity:.4f}，跳過全局驗證")
                best_metrics['is_global_optimal'] = True
                best_metrics['verification_skipped'] = True
                alignment_timing['stage5_global_verification'] = 0.0
                verified_angle = best_angle
                verified_offset_x = best_offset_x
                verified_offset_y = best_offset_y
                verified_similarity = best_similarity
                verification_metrics = {
                    'verification_samples': 0,
                    'verification_improvement': 0.0,
                    'is_global_optimal': True,
                    'verification_best_similarity': best_similarity,
                    'candidates_found': 0,
                    'verification_skipped': True
                }
            elif best_similarity > 0.95:
                # 縮小搜索範圍
                adaptive_rotation_range = min(rotation_range, 5.0)
                adaptive_translation_range = min(10, dynamic_translation_range // 2)
                print(f"階段5：相似度 {best_similarity:.4f}，使用縮小搜索範圍（旋轉±{adaptive_rotation_range:.1f}度，平移±{adaptive_translation_range}像素）")
                verified_angle, verified_offset_x, verified_offset_y, verified_similarity, verification_metrics = \
                    self._global_verification_search(
                        img1_gray, img2_gray, best_angle, best_offset_x, best_offset_y,
                        rotation_range=adaptive_rotation_range, translation_range=adaptive_translation_range
                    )
            else:
                # 使用完整範圍
                print(f"階段5：相似度 {best_similarity:.4f}，使用完整搜索範圍")
                verified_angle, verified_offset_x, verified_offset_y, verified_similarity, verification_metrics = \
                    self._global_verification_search(
                        img1_gray, img2_gray, best_angle, best_offset_x, best_offset_y,
                        rotation_range=rotation_range, translation_range=dynamic_translation_range
                    )
            
            # 如果驗證階段找到更好的解，使用它
            if verified_similarity > best_similarity:
                improvement = verified_similarity - best_similarity
                best_angle = verified_angle
                best_offset_x = verified_offset_x
                best_offset_y = verified_offset_y
                best_similarity = verified_similarity
                best_metrics['verification_improvement'] = float(improvement)
                best_metrics['is_global_optimal'] = False
                print(f"階段5全局驗證：找到更好的解，改進: {improvement:.4f}")
            else:
                best_metrics['is_global_optimal'] = True
                print(f"階段5全局驗證：確認當前解為全局最優")
            
            best_metrics.update(verification_metrics)
        except Exception as e:
            print(f"警告：全局驗證失敗: {str(e)}")
            import traceback
            traceback.print_exc()
            best_metrics['is_global_optimal'] = None
            best_metrics['verification_error'] = str(e)
        
        alignment_timing['stage5_global_verification'] = time.time() - stage5_start
        
        # 記錄最終參數和搜索精度
        best_metrics['final_angle'] = float(best_angle)
        best_metrics['final_offset_x'] = int(best_offset_x)
        best_metrics['final_offset_y'] = int(best_offset_y)
        best_metrics['final_similarity'] = float(best_similarity)
        best_metrics['search_precision'] = {
            'rotation': 0.2,  # 度
            'translation': 1   # 像素
        }
        
        print(f"對齊完成：最終角度={best_angle:.2f}度, 最終偏移=({best_offset_x}, {best_offset_y}), "
              f"最終相似度={best_similarity:.4f}, 是否全局最優={best_metrics.get('is_global_optimal', 'Unknown')}")
        
        # 應用最佳變換到原始圖像（保持原始顏色）
        if len(image2.shape) == 3:
            img2_final = image2.copy()
        else:
            img2_final = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

        # ===== 修正裁切：使用 auto-canvas warpAffine，避免負平移/旋轉把內容丟掉 =====
        h1, w1 = image1.shape[:2]
        center = (w2 // 2, h2 // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0).astype(np.float32)
        M[0, 2] += float(best_offset_x)
        M[1, 2] += float(best_offset_y)

        img2_aligned, canvas_info = self._warp_affine_auto_canvas(
            img2_final,
            M,
            include_sizes=[(w1, h1)],
            border_value=(255, 255, 255)
        )
        best_metrics['alignment_canvas'] = canvas_info
        best_metrics['alignment_canvas_mode'] = 'auto_canvas'
        # ===== 修正裁切結束 =====

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
    
    def _estimate_translation_multi_method(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        translation_range: int,
        scale_factor: float
    ) -> Tuple[int, int, float]:
        """
        使用多種模板匹配方法組合估算平移偏移，提高粗調精度
        
        Args:
            img1: 參考圖像（已縮放）
            img2: 待匹配圖像（已旋轉，未平移）
            translation_range: 平移搜索範圍（原始尺寸）
            scale_factor: 圖像縮放比例
            
        Returns:
            (估算的x偏移, 估算的y偏移, 綜合匹配分數)
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 確保模板不大於搜索圖像
        if w2 > w1 or h2 > h1:
            return 0, 0, 0.0
        
        # 計算圖像中心
        center_x1 = w1 // 2
        center_y1 = h1 // 2
        center_x2 = w2 // 2
        center_y2 = h2 // 2
        
        # 使用多種模板匹配方法
        methods = [
            (cv2.TM_CCOEFF_NORMED, 1.0),  # 相關性係數，權重最高
            (cv2.TM_CCORR_NORMED, 0.8),   # 相關性
            (cv2.TM_SQDIFF_NORMED, 0.6)   # 平方差（值越小越好，需要轉換）
        ]
        
        candidates = []
        
        for method, weight in methods:
            try:
                result = cv2.matchTemplate(img1, img2, method)
                
                if method == cv2.TM_SQDIFF_NORMED:
                    # 平方差：值越小越好，轉換為相似度（值越大越好）
                    _, min_val, _, min_loc = cv2.minMaxLoc(result)
                    score = 1.0 - min_val
                    max_loc = min_loc
                else:
                    # 相關性：值越大越好
                    _, max_val, _, max_loc = cv2.minMaxLoc(result)
                    score = float(max_val)
                
                # 計算偏移
                match_center_x = max_loc[0] + center_x2
                match_center_y = max_loc[1] + center_y2
                offset_x_scaled = center_x1 - match_center_x
                offset_y_scaled = center_y1 - match_center_y
                
                # 轉換回原始尺寸
                offset_x = int(offset_x_scaled / scale_factor)
                offset_y = int(offset_y_scaled / scale_factor)
                
                # 限制在搜索範圍內（支持大範圍平移）
                offset_x = max(-translation_range, min(translation_range, offset_x))
                offset_y = max(-translation_range, min(translation_range, offset_y))
                
                # 加權分數
                weighted_score = score * weight
                candidates.append((offset_x, offset_y, weighted_score, score))
            except Exception:
                continue
        
        if not candidates:
            return 0, 0, 0.0
        
        # 選擇最佳候選：優先考慮高分數，如果分數接近則考慮一致性
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # 如果最高分數明顯優於其他，直接使用
        if len(candidates) > 1 and candidates[0][2] > candidates[1][2] * 1.2:
            return candidates[0][0], candidates[0][1], candidates[0][3]
        
        # 否則，尋找多個方法一致同意的候選
        best_candidate = candidates[0]
        consensus_count = 1
        
        for candidate in candidates[1:]:
            # 檢查是否與最佳候選接近（允許±2像素誤差）
            dx_diff = abs(candidate[0] - best_candidate[0])
            dy_diff = abs(candidate[1] - best_candidate[1])
            
            if dx_diff <= 2 and dy_diff <= 2:
                consensus_count += 1
                # 如果多個方法一致，使用加權平均
                if consensus_count >= 2:
                    # 計算加權平均偏移
                    total_weight = sum(c[2] for c in candidates[:consensus_count])
                    avg_x = sum(c[0] * c[2] for c in candidates[:consensus_count]) / total_weight
                    avg_y = sum(c[1] * c[2] for c in candidates[:consensus_count]) / total_weight
                    avg_score = sum(c[3] for c in candidates[:consensus_count]) / consensus_count
                    
                    return int(avg_x), int(avg_y), float(avg_score)
        
        # 如果沒有共識，返回最高分數的候選
        return best_candidate[0], best_candidate[1], best_candidate[3]
    
    def _coarse_translation_search(
        self,
        img1_gray: np.ndarray,
        img2_gray: np.ndarray,
        translation_range: int
    ) -> Tuple[int, int, float]:
        """
        階段1：平移粗調
        
        使用縮小圖像進行多方法模板匹配，快速估算最佳平移值
        
        Args:
            img1_gray: 參考圖像（灰度）
            img2_gray: 待對齊圖像（灰度）
            translation_range: 平移搜索範圍
            
        Returns:
            (最佳x偏移, 最佳y偏移, 置信度分數)
        """
        # 使用縮小圖像進行快速評估
        scale_factor = 0.3
        h1, w1 = img1_gray.shape[:2]
        h2, w2 = img2_gray.shape[:2]
        
        small_h1 = int(h1 * scale_factor)
        small_w1 = int(w1 * scale_factor)
        small_h2 = int(h2 * scale_factor)
        small_w2 = int(w2 * scale_factor)
        
        img1_small = cv2.resize(img1_gray, (small_w1, small_h1), interpolation=cv2.INTER_AREA)
        img2_small = cv2.resize(img2_gray, (small_w2, small_h2), interpolation=cv2.INTER_AREA)
        
        # 使用多方法模板匹配估算平移
        offset_x, offset_y, confidence = self._estimate_translation_multi_method(
            img1_small, img2_small, translation_range, scale_factor
        )
        
        return offset_x, offset_y, confidence
    
    def _coarse_rotation_search(
        self,
        img1_gray: np.ndarray,
        img2_gray: np.ndarray,
        offset_x: int,
        offset_y: int,
        rotation_range: float
    ) -> Tuple[float, float]:
        """
        階段2：旋轉粗調
        
        固定平移值，只搜索旋轉角度，使用縮小圖像快速評估
        
        Args:
            img1_gray: 參考圖像（灰度）
            img2_gray: 待對齊圖像（灰度）
            offset_x: 已確定的x偏移
            offset_y: 已確定的y偏移
            rotation_range: 旋轉角度搜索範圍
            
        Returns:
            (最佳旋轉角度, 最佳相似度)
        """
        # 使用縮小圖像進行快速評估
        scale_factor = 0.3
        h1, w1 = img1_gray.shape[:2]
        h2, w2 = img2_gray.shape[:2]
        
        small_h1 = int(h1 * scale_factor)
        small_w1 = int(w1 * scale_factor)
        small_h2 = int(h2 * scale_factor)
        small_w2 = int(w2 * scale_factor)
        
        img1_small = cv2.resize(img1_gray, (small_w1, small_h1), interpolation=cv2.INTER_AREA)
        img2_small = cv2.resize(img2_gray, (small_w2, small_h2), interpolation=cv2.INTER_AREA)
        
        # 應用平移到縮小圖像
        offset_x_small = int(offset_x * scale_factor)
        offset_y_small = int(offset_y * scale_factor)
        M_trans = np.float32([[1, 0, offset_x_small], [0, 1, offset_y_small]])
        img2_trans_small = cv2.warpAffine(img2_small, M_trans, (small_w2, small_h2),
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=255)
        
        # 搜索旋轉角度（步長3度）
        angle_step = 3.0
        best_angle = 0.0
        best_similarity = 0.0
        
        for angle in np.arange(-rotation_range, rotation_range + angle_step, angle_step):
            if angle < -80 or angle > 80:
                continue
            
            # 應用旋轉
            center_small = (small_w2 // 2, small_h2 // 2)
            M_rot = cv2.getRotationMatrix2D(center_small, angle, 1.0)
            img2_rot_small = cv2.warpAffine(img2_trans_small, M_rot, (small_w2, small_h2),
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=255)
            
            # 快速評估相似度
            similarity = self._fast_rotation_match(img1_small, img2_rot_small)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_angle = angle
        
        return best_angle, best_similarity

    def _joint_coarse_search_multiscale(
        self,
        img1_gray: np.ndarray,
        img2_gray: np.ndarray,
        rotation_range: float,
        translation_range: int,
        pyramid_scales: Tuple[float, ...] = (0.25, 0.5),
        top_k: int = 5
    ) -> Tuple[float, int, int, float, Dict[str, float]]:
        """
        joint-grid 粗搜尋（角度×平移），並使用多尺度(template matching)金字塔強化平移穩定性。

        目的：取代原本「階段1平移粗調 + 階段2旋轉粗調」分離流程，於一次 multi-scale iteration 中
        同時找到合理的 (angle, dx, dy) 初始值，供後續階段3/4/5精細化。

        Args:
            img1_gray: 參考圖像（灰度，完整尺寸）
            img2_gray: 待對齊圖像（灰度，完整尺寸）
            rotation_range: 旋轉角度搜索範圍（度）
            translation_range: 平移搜索範圍（像素，原始尺寸）
            pyramid_scales: 多尺度縮放比例（由粗到細）
            top_k: 每個尺度保留的候選解數量

        Returns:
            (best_angle, best_offset_x, best_offset_y, best_similarity, timing_dict)
        """
        import time

        def _resize_gray(img: np.ndarray, scale: float) -> np.ndarray:
            if scale >= 0.999:
                return img
            h, w = img.shape[:2]
            nh = max(32, int(round(h * scale)))
            nw = max(32, int(round(w * scale)))
            return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        def _bbox_of_ink(img: np.ndarray, thr: int = 245) -> Optional[Tuple[int, int, int, int]]:
            # 以「非白色」視為印面內容，取得 bounding box
            if img is None or img.size == 0:
                return None
            mask = (img < thr).astype(np.uint8)
            if mask.sum() < 10:
                return None
            ys, xs = np.where(mask > 0)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            if x1 - x0 < 5 or y1 - y0 < 5:
                return None
            return x0, y0, x1 + 1, y1 + 1

        timing: Dict[str, float] = {}
        overall_start = time.time()

        # 候選格式: (angle, dx, dy, similarity)
        candidates: list[Tuple[float, int, int, float]] = [(0.0, 0, 0, -1.0)]

        prev_angle_step: Optional[float] = None

        for scale_idx, scale in enumerate(pyramid_scales):
            scale_start = time.time()

            img1_s = _resize_gray(img1_gray, scale)
            img2_s = _resize_gray(img2_gray, scale)
            h1s, w1s = img1_s.shape[:2]
            h2s, w2s = img2_s.shape[:2]
            center_s = (w2s // 2, h2s // 2)

            # 自適應角度步長：粗尺度用較大步長，細尺度減半
            if prev_angle_step is None:
                # 以 rotation_range 推估合理粗步長，避免過多角度
                prev_angle_step = 3.0 if rotation_range >= 12.0 else 2.0
            else:
                prev_angle_step = max(1.0, prev_angle_step / 2.0)
            angle_step = prev_angle_step

            # 限制平移範圍於縮放後的像素
            tr_s = max(2, int(round(translation_range * scale)))

            # 準備角度候選集合（第一層全掃，後續只在前一層 top_k 附近掃）
            angle_set = set()
            if scale_idx == 0:
                for a in np.arange(-rotation_range, rotation_range + angle_step, angle_step):
                    if -80.0 <= float(a) <= 80.0:
                        angle_set.add(float(round(float(a), 3)))
            else:
                # 局部窗：依照前一層候選的角度，做小範圍掃描
                local_window = max(angle_step * 2.0, min(6.0, rotation_range * 0.25))
                for (a0, _, _, _) in candidates[:top_k]:
                    for a in np.arange(a0 - local_window, a0 + local_window + angle_step, angle_step):
                        if -80.0 <= float(a) <= 80.0:
                            angle_set.add(float(round(float(a), 3)))

            angle_list = sorted(angle_set)

            # 旋轉緩存（每個尺度一份）
            rotation_cache: Dict[float, np.ndarray] = {}

            found: list[Tuple[float, int, int, float]] = []

            for angle in angle_list:
                if angle in rotation_cache:
                    img2_rot = rotation_cache[angle]
                else:
                    M_rot = cv2.getRotationMatrix2D(center_s, angle, 1.0)
                    img2_rot = cv2.warpAffine(
                        img2_s, M_rot, (w2s, h2s),
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=255
                    )
                    rotation_cache[angle] = img2_rot

                # 從旋轉後的 img2 中擷取印面 bbox 作為 template（確保 template 小於 search image）
                bbox = _bbox_of_ink(img2_rot)
                if bbox is None:
                    continue
                x0, y0, x1, y1 = bbox
                pad = 4
                x0p = max(0, x0 - pad)
                y0p = max(0, y0 - pad)
                x1p = min(w2s, x1 + pad)
                y1p = min(h2s, y1 + pad)
                template = img2_rot[y0p:y1p, x0p:x1p]

                # 如果 template 太大（等於或大於 img1），縮小到中心區域避免 matchTemplate 退化成 1x1
                th, tw = template.shape[:2]
                if tw >= w1s or th >= h1s:
                    cx0 = int(round(w2s * 0.15))
                    cy0 = int(round(h2s * 0.15))
                    cx1 = int(round(w2s * 0.85))
                    cy1 = int(round(h2s * 0.85))
                    template = img2_rot[cy0:cy1, cx0:cx1]
                    x0p, y0p = cx0, cy0
                    th, tw = template.shape[:2]
                    if tw >= w1s or th >= h1s:
                        continue

                # 在 img1 上做 template matching 找出 img2 的最佳平移（top-left 對齊）
                try:
                    res = cv2.matchTemplate(img1_s, template, cv2.TM_CCOEFF_NORMED)
                    _, _, _, max_loc = cv2.minMaxLoc(res)
                except Exception:
                    continue

                dx_s = int(max_loc[0] - x0p)
                dy_s = int(max_loc[1] - y0p)

                # 將縮放座標轉回原始座標，並限制範圍
                dx = int(round(dx_s / scale))
                dy = int(round(dy_s / scale))
                dx = max(-translation_range, min(translation_range, dx))
                dy = max(-translation_range, min(translation_range, dy))

                # 為了粗搜尋成本，先做縮放後的快速相似度評估
                dx_s_clamped = max(-tr_s, min(tr_s, int(round(dx * scale))))
                dy_s_clamped = max(-tr_s, min(tr_s, int(round(dy * scale))))
                M_trans = np.float32([[1, 0, float(dx_s_clamped)], [0, 1, float(dy_s_clamped)]])
                img2_trans = cv2.warpAffine(
                    img2_rot, M_trans, (w2s, h2s),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255
                )

                sim = self._fast_rotation_match(img1_s, img2_trans)
                found.append((float(angle), int(dx), int(dy), float(sim)))

            # 選出本尺度 top_k
            if found:
                found.sort(key=lambda x: x[3], reverse=True)
                candidates = found[:max(1, top_k)]
            else:
                # 如果完全找不到候選，保留上一輪結果
                candidates = candidates[:1]

            timing[f'stage12_scale_{scale}'] = time.time() - scale_start

        # 最終選擇最佳候選
        best_angle, best_dx, best_dy, best_sim = candidates[0]
        timing['stage12_joint_coarse_total'] = time.time() - overall_start
        # 供上層做 low-overlap multi-start（僅保留少量候選，避免 payload 過大）
        try:
            timing['stage12_candidates'] = [
                {'angle': float(a), 'dx': int(dx), 'dy': int(dy), 'sim': float(s)}
                for (a, dx, dy, s) in candidates[:max(1, min(top_k, len(candidates)))]
            ]
        except Exception:
            timing['stage12_candidates'] = []
        return best_angle, best_dx, best_dy, best_sim, timing

    def _fast_overlap_ratio(
        self,
        img1_gray: np.ndarray,
        img2_gray: np.ndarray,
        angle: float,
        offset_x: int,
        offset_y: int,
        scale: float = 0.3,
        white_threshold: int = 245
    ) -> float:
        """
        以縮小圖快速估計 overlap_ratio（印面重疊率），用於偵測「平移落錯峰」。

        定義：
        - mask = (gray < white_threshold) 視為印面內容
        - overlap_ratio = overlap_pixels / union_pixels

        注意：此指標目的是「快速方向性判斷」，不是最終精度指標。
        """
        if img1_gray is None or img2_gray is None or img1_gray.size == 0 or img2_gray.size == 0:
            return 0.0

        # resize（統一以 img1 的座標系計算 overlap，避免因尺寸差/畫布差導致誤判）
        h1, w1 = img1_gray.shape[:2]
        h2, w2 = img2_gray.shape[:2]
        nh1 = max(32, int(round(h1 * scale)))
        nw1 = max(32, int(round(w1 * scale)))
        nh2 = max(32, int(round(h2 * scale)))
        nw2 = max(32, int(round(w2 * scale)))
        img1_s = cv2.resize(img1_gray, (nw1, nh1), interpolation=cv2.INTER_AREA)
        img2_s = cv2.resize(img2_gray, (nw2, nh2), interpolation=cv2.INTER_AREA)

        # rotate + translate img2_s（先在 img2 自己的座標系旋轉，再以 img1 尺寸作為輸出畫布）
        center = (nw2 // 2, nh2 // 2)
        M_rot = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        img2_rot = cv2.warpAffine(
            img2_s, M_rot, (nw2, nh2),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255
        )

        dx_s = int(round(offset_x * scale))
        dy_s = int(round(offset_y * scale))
        M_trans = np.float32([[1, 0, float(dx_s)], [0, 1, float(dy_s)]])
        img2_t = cv2.warpAffine(
            img2_rot, M_trans, (nw2, nh2),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255
        )

        # 將 img2_t 放到 img1 的畫布尺寸（左上角對齊，與 overlay.py 的 padding 規則一致）
        if img2_t.shape[0] != nh1 or img2_t.shape[1] != nw1:
            canvas = np.full((nh1, nw1), 255, dtype=img2_t.dtype)
            hh = min(nh1, img2_t.shape[0])
            ww = min(nw1, img2_t.shape[1])
            canvas[:hh, :ww] = img2_t[:hh, :ww]
            img2_t = canvas

        # build masks（同尺寸）
        mask1 = img1_s < white_threshold
        mask2 = img2_t < white_threshold
        union = mask1 | mask2
        union_pixels = int(np.sum(union))
        if union_pixels <= 0:
            return 0.0
        overlap = mask1 & mask2
        overlap_pixels = int(np.sum(overlap))
        return float(overlap_pixels / union_pixels)

    def _translation_rescue_search(
        self,
        img1_gray: np.ndarray,
        img2_gray: np.ndarray,
        angle: float,
        initial_offset_x: int,
        initial_offset_y: int,
        overlap_threshold: float = 0.6,
        radius: int = 120,
        steps: Tuple[int, ...] = (8, 3, 1),
        scale: float = 0.3
    ) -> Tuple[int, int, float, float, Dict[str, float]]:
        """
        平移救援搜尋（固定 angle），用 overlap_ratio 作為評分，專治「大錯峰」。

        策略（多段步長）：radius=±120，step=8→3→1，每段以上一段最佳點為中心縮小窗口。

        Returns:
            (best_offset_x, best_offset_y, overlap_before, overlap_after, timing)
        """
        import time
        timing: Dict[str, float] = {}
        start = time.time()

        overlap_before = self._fast_overlap_ratio(
            img1_gray, img2_gray, angle, initial_offset_x, initial_offset_y, scale=scale
        )
        timing['stage3_translation_rescue_evals'] = 0.0
        best_x = int(initial_offset_x)
        best_y = int(initial_offset_y)
        best_overlap = float(overlap_before)

        if best_overlap >= overlap_threshold:
            timing['stage3_translation_rescue_total'] = time.time() - start
            return best_x, best_y, overlap_before, best_overlap, timing

        # 每段縮小窗口：第一段 full radius，之後縮到 radius/3、radius/12（與步長對應）
        window_by_step = {
            8: radius,
            3: max(40, radius // 3),
            1: 10
        }

        evals = 0
        for step in steps:
            seg_start = time.time()
            win = window_by_step.get(step, max(10, radius // 3))

            # grid search around current best
            for dx in range(-win, win + 1, step):
                ox = best_x + dx
                for dy in range(-win, win + 1, step):
                    oy = best_y + dy
                    evals += 1
                    ov = self._fast_overlap_ratio(img1_gray, img2_gray, angle, ox, oy, scale=scale)
                    if ov > best_overlap:
                        best_overlap = ov
                        best_x = ox
                        best_y = oy

            timing[f'stage3_translation_rescue_step_{step}'] = time.time() - seg_start

        timing['stage3_translation_rescue_evals'] = float(evals)
        timing['stage3_translation_rescue_total'] = time.time() - start
        return best_x, best_y, overlap_before, best_overlap, timing

    def _angle_sign_disambiguation(
        self,
        img1_gray: np.ndarray,
        img2_gray: np.ndarray,
        angle: float,
        offset_x: int,
        offset_y: int,
        scale: float = 0.3,
        improve_threshold: float = 0.05,
        rescue_radius: int = 60,
        rescue_steps: Tuple[int, ...] = (6, 2, 1)
    ) -> Tuple[float, int, int, Dict[str, Any]]:
        """
        解決 +θ/-θ 二義性：用 overlap_ratio 快速判斷角度正負，必要時在 flip 後做小預算平移救援。

        Returns:
            (best_angle, best_dx, best_dy, metrics)
        """
        metrics: Dict[str, Any] = {
            'angle_sign_check_triggered': False,
            'angle_sign_flipped': False
        }

        angle0 = float(angle)
        angle1 = -float(angle)

        try:
            ov0_raw = float(self._fast_overlap_ratio(img1_gray, img2_gray, angle0, offset_x, offset_y, scale=scale))
            ov1_raw = float(self._fast_overlap_ratio(img1_gray, img2_gray, angle1, offset_x, offset_y, scale=scale))
        except Exception as e:
            metrics['angle_sign_check_error'] = str(e)
            return angle0, int(offset_x), int(offset_y), metrics

        metrics['angle_sign_check_triggered'] = True
        metrics['overlap_before_sign_check'] = ov0_raw
        metrics['overlap_flip_raw'] = ov1_raw

        # 關鍵改動：不再只看 raw overlap（固定 dx/dy），而是對 +θ/-θ 都做小預算平移微搜尋後再比較
        # 這專治「翻號後需要小幅平移才能變好」的情況。
        def eval_with_micro_rescue(a: float) -> Tuple[int, int, float, Dict[str, Any]]:
            rx, ry, ob, oa, rescue_timing = self._translation_rescue_search(
                img1_gray,
                img2_gray,
                angle=a,
                initial_offset_x=int(offset_x),
                initial_offset_y=int(offset_y),
                overlap_threshold=1.0,  # 幾乎不早退，公平比較
                radius=int(rescue_radius),
                steps=tuple(rescue_steps),
                scale=scale
            )
            extra = {
                'overlap_before': float(ob),
                'overlap_after': float(oa),
                'evals': rescue_timing.get('stage3_translation_rescue_evals', 0.0) if rescue_timing else 0.0
            }
            return int(rx), int(ry), float(oa), extra

        # 只有在 overlap 低時才值得花這點 budget（呼叫端已經做過 gating，但這裡也保護一下）
        if max(ov0_raw, ov1_raw) >= 0.85:
            metrics['overlap_after_sign_check'] = ov0_raw
            return angle0, int(offset_x), int(offset_y), metrics

        try:
            x0, y0, ov0_best, ex0 = eval_with_micro_rescue(angle0)
            x1, y1, ov1_best, ex1 = eval_with_micro_rescue(angle1)
        except Exception as e:
            metrics['angle_sign_micro_rescue_error'] = str(e)
            metrics['overlap_after_sign_check'] = ov0_raw
            return angle0, int(offset_x), int(offset_y), metrics

        metrics['overlap_after_sign_check'] = float(ov0_best)
        metrics['overlap_flip_after_micro_rescue'] = float(ov1_best)
        metrics['angle_sign_micro_rescue_evals'] = float(ex0.get('evals', 0.0) + ex1.get('evals', 0.0))

        # flip if -θ 明顯更好
        if ov1_best > ov0_best + improve_threshold:
            metrics['angle_sign_flipped'] = True
            metrics['overlap_after_sign_check'] = float(ov1_best)
            return float(angle1), int(x1), int(y1), metrics

        return float(angle0), int(x0), int(y0), metrics

    def _warp_affine_auto_canvas(
        self,
        img: np.ndarray,
        M: np.ndarray,
        include_sizes: Optional[List[Tuple[int, int]]] = None,
        border_value: Optional[Tuple[int, int, int]] = None
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        針對 warpAffine 的裁切問題：自動擴張輸出畫布，確保變換後內容不會被丟掉。

        - 以變換後的四角點 bounding box 計算輸出大小
        - 額外把 include_sizes（例如 reference image 的 (w,h)）也納入 bounding box，
          讓兩張圖可以被放進同一畫布（同一座標系）

        Returns:
            warped_img, canvas_info = {'shift_x','shift_y','w','h'}
        """
        if include_sizes is None:
            include_sizes = []

        if img is None or img.size == 0:
            return img, {'shift_x': 0, 'shift_y': 0, 'w': 0, 'h': 0}

        h, w = img.shape[:2]
        M = np.asarray(M, dtype=np.float32)

        # 原圖四角（img 的座標系）
        src_pts = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float32).reshape(1, -1, 2)
        warped_pts = cv2.transform(src_pts, M).reshape(-1, 2)

        all_pts = [warped_pts]
        # 參考圖四角（目標座標系，identity）
        for (iw, ih) in include_sizes:
            iw = int(iw)
            ih = int(ih)
            if iw <= 0 or ih <= 0:
                continue
            ref_pts = np.array([[0, 0], [iw, 0], [0, ih], [iw, ih]], dtype=np.float32)
            all_pts.append(ref_pts)

        all_pts = np.vstack(all_pts)
        min_x = float(np.floor(np.min(all_pts[:, 0])))
        min_y = float(np.floor(np.min(all_pts[:, 1])))
        max_x = float(np.ceil(np.max(all_pts[:, 0])))
        max_y = float(np.ceil(np.max(all_pts[:, 1])))

        # 因為 include_sizes 會包含 (0,0)，所以 min_x/min_y 應 <= 0，shift 會 >= 0
        shift_x = int(-min_x) if min_x < 0 else 0
        shift_y = int(-min_y) if min_y < 0 else 0

        out_w = int(max(1, max_x - min_x))
        out_h = int(max(1, max_y - min_y))

        M_adj = M.copy()
        M_adj[0, 2] += float(shift_x)
        M_adj[1, 2] += float(shift_y)

        if border_value is None:
            # 如果是灰階圖，OpenCV 也接受單一 int；這裡統一用白色
            border_value = (255, 255, 255) if (img.ndim == 3 and img.shape[2] == 3) else (255, 255, 255)

        warped = cv2.warpAffine(
            img,
            M_adj,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value
        )

        return warped, {'shift_x': int(shift_x), 'shift_y': int(shift_y), 'w': int(out_w), 'h': int(out_h)}
    
    def _fine_translation_search(
        self,
        img1_gray: np.ndarray,
        img2_gray: np.ndarray,
        angle: float,
        initial_offset_x: int,
        initial_offset_y: int,
        search_range: int = 8
    ) -> Tuple[int, int, float]:
        """
        階段3：平移細調
        
        在完整尺寸圖像上，應用最佳旋轉角度後，精細調整平移
        
        Args:
            img1_gray: 參考圖像（完整尺寸灰度）
            img2_gray: 待對齊圖像（完整尺寸灰度）
            angle: 已確定的旋轉角度
            initial_offset_x: 初始x偏移
            initial_offset_y: 初始y偏移
            search_range: 搜索範圍（像素）
            
        Returns:
            (精細x偏移, 精細y偏移, 最佳相似度)
        """
        h2, w2 = img2_gray.shape[:2]
        
        # 應用旋轉
        center = (w2 // 2, h2 // 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        img2_rot = cv2.warpAffine(img2_gray, M_rot, (w2, h2),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255)
        
        # 在初始偏移附近進行精細搜索
        best_offset_x = initial_offset_x
        best_offset_y = initial_offset_y
        best_similarity = 0.0
        
        for dx in range(-search_range, search_range + 1):
            for dy in range(-search_range, search_range + 1):
                offset_x = initial_offset_x + dx
                offset_y = initial_offset_y + dy
                
                # 應用平移
                M_trans = np.float32([[1, 0, float(offset_x)], [0, 1, float(offset_y)]])
                img2_transformed = cv2.warpAffine(img2_rot, M_trans, (w2, h2),
                                                 borderMode=cv2.BORDER_CONSTANT,
                                                 borderValue=255)
                
                # 計算相似度
                similarity = self._fast_rotation_match(img1_gray, img2_transformed)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_offset_x = offset_x
                    best_offset_y = offset_y
        
        return best_offset_x, best_offset_y, best_similarity
    
    def _alternating_fine_tuning(
        self,
        img1_gray: np.ndarray,
        img2_gray: np.ndarray,
        initial_angle: float,
        initial_offset_x: int,
        initial_offset_y: int,
        rotation_range: float = 2.0,
        translation_range: int = 3,
        iterations: int = 2
    ) -> Tuple[float, int, int, float, Dict[str, float]]:
        """
        階段4：旋轉細調與平移細調（交替優化）
        
        交替優化旋轉角度和平移，重複指定次數
        
        Args:
            img1_gray: 參考圖像（完整尺寸灰度）
            img2_gray: 待對齊圖像（完整尺寸灰度）
            initial_angle: 初始旋轉角度
            initial_offset_x: 初始x偏移
            initial_offset_y: 初始y偏移
            rotation_range: 旋轉搜索範圍（度）
            translation_range: 平移搜索範圍（像素）
            iterations: 交替優化次數
            
        Returns:
            (最佳旋轉角度, 最佳x偏移, 最佳y偏移, 最佳相似度, 時間詳情字典)
        """
        import time
        timing = {}
        
        best_angle = initial_angle
        best_offset_x = initial_offset_x
        best_offset_y = initial_offset_y
        best_similarity = 0.0
        
        h2, w2 = img2_gray.shape[:2]
        
        # 初始評估
        center = (w2 // 2, h2 // 2)
        M_rot = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        img2_rot = cv2.warpAffine(img2_gray, M_rot, (w2, h2),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255)
        M_trans = np.float32([[1, 0, float(best_offset_x)], [0, 1, float(best_offset_y)]])
        img2_transformed = cv2.warpAffine(img2_rot, M_trans, (w2, h2),
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=255)
        best_similarity = self._fast_rotation_match(img1_gray, img2_transformed)
        
        rotation_time_total = 0.0
        translation_time_total = 0.0
        
        # 優化：緩存旋轉變換結果
        rotation_cache = {}
        
        # 交替優化
        for iteration in range(iterations):
            # 4a. 旋轉細調：在最佳平移下，搜索旋轉角度
            rotation_start = time.time()
            # 使用0.2度步長以達到更高精度
            angle_range = np.arange(-rotation_range, rotation_range + 0.2, 0.2)
            
            for angle_offset in angle_range:
                angle = best_angle + angle_offset
                if angle < -80 or angle > 80:
                    continue
                
                # 應用旋轉（使用緩存）
                if angle in rotation_cache:
                    img2_rot = rotation_cache[angle]
                else:
                    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                    img2_rot = cv2.warpAffine(img2_gray, M_rot, (w2, h2),
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=255)
                    rotation_cache[angle] = img2_rot
                
                # 應用當前最佳平移
                M_trans = np.float32([[1, 0, float(best_offset_x)], [0, 1, float(best_offset_y)]])
                img2_transformed = cv2.warpAffine(img2_rot, M_trans, (w2, h2),
                                                 borderMode=cv2.BORDER_CONSTANT,
                                                 borderValue=255)
                
                # 計算相似度
                similarity = self._fast_rotation_match(img1_gray, img2_transformed)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_angle = angle
            
            rotation_time_total += time.time() - rotation_start
            
            # 4b. 平移細調：在最佳旋轉下，搜索平移
            translation_start = time.time()
            
            # 應用最佳旋轉（使用緩存）
            if best_angle in rotation_cache:
                img2_rot = rotation_cache[best_angle]
            else:
                M_rot = cv2.getRotationMatrix2D(center, best_angle, 1.0)
                img2_rot = cv2.warpAffine(img2_gray, M_rot, (w2, h2),
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=255)
                rotation_cache[best_angle] = img2_rot
            
            for dx in range(-translation_range, translation_range + 1):
                for dy in range(-translation_range, translation_range + 1):
                    offset_x = best_offset_x + dx
                    offset_y = best_offset_y + dy
                    
                    # 應用平移
                    M_trans = np.float32([[1, 0, float(offset_x)], [0, 1, float(offset_y)]])
                    img2_transformed = cv2.warpAffine(img2_rot, M_trans, (w2, h2),
                                                     borderMode=cv2.BORDER_CONSTANT,
                                                     borderValue=255)
                    
                    # 計算相似度
                    similarity = self._fast_rotation_match(img1_gray, img2_transformed)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_offset_x = offset_x
                        best_offset_y = offset_y
            
            translation_time_total += time.time() - translation_start
        
        timing['stage4_rotation_fine'] = rotation_time_total
        timing['stage4_translation_fine'] = translation_time_total
        timing['stage4_total'] = rotation_time_total + translation_time_total
        
        return best_angle, best_offset_x, best_offset_y, best_similarity, timing
    
    def _global_verification_search(
        self,
        img1_gray: np.ndarray,
        img2_gray: np.ndarray,
        best_angle: float,
        best_offset_x: int,
        best_offset_y: int,
        rotation_range: float,
        translation_range: int
    ) -> Tuple[float, int, int, float, Dict[str, Any]]:
        """
        全局驗證搜索：在整個搜索空間進行稀疏採樣，確認沒有遺漏更好的解
        
        策略：
        1. 在最佳參數附近進行稀疏採樣（旋轉每1度，平移每2像素）
        2. 如果發現更好的候選，在該候選附近進行精細搜索（旋轉0.2度，平移1像素）
        3. 返回最佳結果和驗證指標
        
        Args:
            img1_gray: 參考圖像（完整尺寸灰度）
            img2_gray: 待對齊圖像（完整尺寸灰度）
            best_angle: 當前最佳旋轉角度
            best_offset_x: 當前最佳x偏移
            best_offset_y: 當前最佳y偏移
            rotation_range: 旋轉搜索範圍（度）
            translation_range: 平移搜索範圍（像素）
            
        Returns:
            (最佳旋轉角度, 最佳x偏移, 最佳y偏移, 最佳相似度, 驗證指標字典)
        """
        import time
        verification_start = time.time()
        metrics = {
            'verification_samples': 0,
            'verification_improvement': 0.0,
            'is_global_optimal': True,
            'verification_best_similarity': 0.0,
            'candidates_found': 0
        }
        
        h2, w2 = img2_gray.shape[:2]
        center = (w2 // 2, h2 // 2)
        
        # 初始評估當前最佳解
        M_rot = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        img2_rot = cv2.warpAffine(img2_gray, M_rot, (w2, h2),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255)
        M_trans = np.float32([[1, 0, float(best_offset_x)], [0, 1, float(best_offset_y)]])
        img2_transformed = cv2.warpAffine(img2_rot, M_trans, (w2, h2),
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=255)
        current_best_similarity = self._fast_rotation_match(img1_gray, img2_transformed)
        metrics['verification_best_similarity'] = current_best_similarity
        
        verified_angle = best_angle
        verified_offset_x = best_offset_x
        verified_offset_y = best_offset_y
        verified_similarity = current_best_similarity
        
        # 階段1：稀疏採樣驗證（在最佳參數附近進行廣泛搜索）
        # 旋轉：在最佳角度±rotation_range範圍內，每1度採樣
        # 平移：在最佳平移±10像素範圍內，每2像素採樣
        sparse_rotation_range = min(rotation_range, 10.0)  # 限制驗證範圍，避免過大
        sparse_translation_range = min(10, translation_range // 4)  # 限制平移驗證範圍
        
        candidates = []  # 存儲候選解
        
        # 優化：提前終止機制和旋轉緩存
        no_improvement_count = 0
        max_no_improvement = 20  # 連續20個採樣無改進則提前終止
        early_terminated = False
        rotation_cache = {}  # 緩存旋轉變換結果
        
        for angle_offset in np.arange(-sparse_rotation_range, sparse_rotation_range + 1.0, 1.0):
            if early_terminated:
                break
                
            angle = best_angle + angle_offset
            if angle < -80 or angle > 80:
                continue
            
            for dx in range(-sparse_translation_range, sparse_translation_range + 1, 2):
                if early_terminated:
                    break
                    
                for dy in range(-sparse_translation_range, sparse_translation_range + 1, 2):
                    offset_x = best_offset_x + dx
                    offset_y = best_offset_y + dy
                    
                    metrics['verification_samples'] += 1
                    
                    # 應用變換（使用緩存）
                    if angle in rotation_cache:
                        img2_rot = rotation_cache[angle]
                    else:
                        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                        img2_rot = cv2.warpAffine(img2_gray, M_rot, (w2, h2),
                                                 borderMode=cv2.BORDER_CONSTANT,
                                                 borderValue=255)
                        rotation_cache[angle] = img2_rot
                    
                    M_trans = np.float32([[1, 0, float(offset_x)], [0, 1, float(offset_y)]])
                    img2_transformed = cv2.warpAffine(img2_rot, M_trans, (w2, h2),
                                                     borderMode=cv2.BORDER_CONSTANT,
                                                     borderValue=255)
                    
                    # 計算相似度
                    similarity = self._fast_rotation_match(img1_gray, img2_transformed)
                    
                    # 如果發現更好的候選（相似度提升超過0.01），記錄它
                    if similarity > current_best_similarity + 0.01:
                        candidates.append((angle, offset_x, offset_y, similarity))
                        metrics['candidates_found'] += 1
                        no_improvement_count = 0  # 重置計數器
                    else:
                        no_improvement_count += 1
                        if no_improvement_count >= max_no_improvement:
                            # 提前終止稀疏採樣
                            early_terminated = True
                            print(f"全局驗證：連續 {max_no_improvement} 個採樣無改進，提前終止稀疏採樣")
                            break
        
        # 階段2：對候選解進行精細搜索
        # 如果發現候選解，在每個候選解附近進行精細搜索（0.2度旋轉，1像素平移）
        # 優化：只對相似度提升>0.02 的候選進行精細搜索
        if candidates:
            # 選擇最佳候選解
            candidates.sort(key=lambda x: x[3], reverse=True)
            best_candidate = candidates[0]
            candidate_angle, candidate_offset_x, candidate_offset_y, candidate_similarity = best_candidate
            
            # 只對相似度提升>0.02 的候選進行精細搜索
            improvement_threshold = 0.02
            if candidate_similarity - current_best_similarity > improvement_threshold:
                # 在候選解附近進行精細搜索
                fine_rotation_range = 0.5  # ±0.5度
                fine_translation_range = 2  # ±2像素
                
                for angle_offset in np.arange(-fine_rotation_range, fine_rotation_range + 0.2, 0.2):
                    angle = candidate_angle + angle_offset
                    if angle < -80 or angle > 80:
                        continue
                    
                    for dx in range(-fine_translation_range, fine_translation_range + 1):
                        for dy in range(-fine_translation_range, fine_translation_range + 1):
                            offset_x = candidate_offset_x + dx
                            offset_y = candidate_offset_y + dy
                            
                            metrics['verification_samples'] += 1
                            
                            # 應用變換（使用緩存）
                            if angle in rotation_cache:
                                img2_rot = rotation_cache[angle]
                            else:
                                M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                                img2_rot = cv2.warpAffine(img2_gray, M_rot, (w2, h2),
                                                         borderMode=cv2.BORDER_CONSTANT,
                                                         borderValue=255)
                                rotation_cache[angle] = img2_rot
                            
                            M_trans = np.float32([[1, 0, float(offset_x)], [0, 1, float(offset_y)]])
                            img2_transformed = cv2.warpAffine(img2_rot, M_trans, (w2, h2),
                                                             borderMode=cv2.BORDER_CONSTANT,
                                                             borderValue=255)
                            
                            # 計算相似度
                            similarity = self._fast_rotation_match(img1_gray, img2_transformed)
                            
                            if similarity > verified_similarity:
                                verified_similarity = similarity
                                verified_angle = angle
                                verified_offset_x = offset_x
                                verified_offset_y = offset_y
                
                # 計算改進幅度
                improvement = verified_similarity - current_best_similarity
                metrics['verification_improvement'] = improvement
                metrics['verification_best_similarity'] = verified_similarity
                
                if improvement > 0.001:  # 如果改進超過0.1%，認為找到了更好的解
                    metrics['is_global_optimal'] = False
                    print(f"全局驗證：發現更好的解，改進幅度: {improvement:.4f}, "
                          f"新角度: {verified_angle:.2f}度, 新偏移: ({verified_offset_x}, {verified_offset_y})")
                else:
                    metrics['is_global_optimal'] = True
                    print(f"全局驗證：確認當前解為全局最優，相似度: {verified_similarity:.4f}")
            else:
                # 候選解改進幅度不夠，跳過精細搜索
                verified_angle = candidate_angle
                verified_offset_x = candidate_offset_x
                verified_offset_y = candidate_offset_y
                verified_similarity = candidate_similarity
                improvement = verified_similarity - current_best_similarity
                metrics['verification_improvement'] = improvement
                metrics['verification_best_similarity'] = verified_similarity
                metrics['fine_search_skipped'] = True
                print(f"全局驗證：候選解改進幅度 {improvement:.4f} 未達精細搜索閾值 {improvement_threshold:.2f}，跳過精細搜索")
        else:
            # 沒有發現更好的候選解，確認當前解為全局最優
            metrics['is_global_optimal'] = True
            print(f"全局驗證：未發現更好的候選解，確認當前解為全局最優，相似度: {verified_similarity:.4f}")
        
        metrics['verification_time'] = time.time() - verification_start
        
        return verified_angle, verified_offset_x, verified_offset_y, verified_similarity, metrics
    
    def _refine_translation_coarse(
        self,
        img1_gray: np.ndarray,
        img2_gray: np.ndarray,
        best_angle: float,
        initial_offset_x: int,
        initial_offset_y: int,
        translation_range: int
    ) -> Tuple[int, int, float]:
        """
        階段1.5：使用中等尺寸圖像進行平移粗調優化
        
        Args:
            img1_gray: 參考圖像（灰度）
            img2_gray: 待對齊圖像（灰度）
            best_angle: 階段1找到的最佳角度
            initial_offset_x: 階段1估算的x偏移
            initial_offset_y: 階段1估算的y偏移
            translation_range: 平移搜索範圍
            
        Returns:
            (優化後的x偏移, 優化後的y偏移, 置信度分數)
        """
        # 使用中等尺寸圖像（scale_factor=0.6）進行更精確的平移估算
        scale_factor = 0.6
        h1, w1 = img1_gray.shape[:2]
        h2, w2 = img2_gray.shape[:2]
        
        medium_h1 = int(h1 * scale_factor)
        medium_w1 = int(w1 * scale_factor)
        medium_h2 = int(h2 * scale_factor)
        medium_w2 = int(w2 * scale_factor)
        
        img1_medium = cv2.resize(img1_gray, (medium_w1, medium_h1), interpolation=cv2.INTER_AREA)
        img2_medium = cv2.resize(img2_gray, (medium_w2, medium_h2), interpolation=cv2.INTER_AREA)
        
        # 應用最佳旋轉角度到中等尺寸圖像
        center_medium = (medium_w2 // 2, medium_h2 // 2)
        M_rot = cv2.getRotationMatrix2D(center_medium, best_angle, 1.0)
        img2_rot_medium = cv2.warpAffine(img2_medium, M_rot, (medium_w2, medium_h2),
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=255)
        
        # 使用多方法模板匹配估算平移
        medium_translation_range = int(translation_range * scale_factor)
        offset_x_medium, offset_y_medium, confidence = self._estimate_translation_multi_method(
            img1_medium, img2_rot_medium, medium_translation_range, scale_factor
        )
        
        # 轉換回原始尺寸
        offset_x_refined = int(offset_x_medium / scale_factor)
        offset_y_refined = int(offset_y_medium / scale_factor)
        
        # 限制在搜索範圍內
        offset_x_refined = max(-translation_range, min(translation_range, offset_x_refined))
        offset_y_refined = max(-translation_range, min(translation_range, offset_y_refined))
        
        return offset_x_refined, offset_y_refined, confidence
    
    def _refine_translation_fine(
        self,
        img1_gray: np.ndarray,
        img2_gray: np.ndarray,
        angle: float,
        initial_offset_x: int,
        initial_offset_y: int,
        search_range: int = 5
    ) -> Tuple[int, int, float]:
        """
        階段2中對每個候選角度進行平移優化
        
        Args:
            img1_gray: 參考圖像（完整尺寸灰度）
            img2_gray: 待對齊圖像（完整尺寸灰度）
            angle: 當前候選角度
            initial_offset_x: 初始x偏移（來自階段1.5）
            initial_offset_y: 初始y偏移（來自階段1.5）
            search_range: 搜索範圍（像素）
            
        Returns:
            (優化後的x偏移, 優化後的y偏移, 置信度分數)
        """
        h2, w2 = img2_gray.shape[:2]
        
        # 應用旋轉
        center = (w2 // 2, h2 // 2)
        M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
        img2_rot = cv2.warpAffine(img2_gray, M_rot, (w2, h2),
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255)
        
        # 使用模板匹配在初始偏移附近進行精細搜索
        # 搜索範圍：initial_offset ± search_range，步長1像素
        best_offset_x = initial_offset_x
        best_offset_y = initial_offset_y
        best_score = 0.0
        
        # 在初始偏移附近進行小範圍搜索
        for dx in range(-search_range, search_range + 1):
            for dy in range(-search_range, search_range + 1):
                offset_x = initial_offset_x + dx
                offset_y = initial_offset_y + dy
                
                # 應用平移
                M_trans = np.float32([[1, 0, float(offset_x)], [0, 1, float(offset_y)]])
                img2_transformed = cv2.warpAffine(img2_rot, M_trans, (w2, h2),
                                                 borderMode=cv2.BORDER_CONSTANT,
                                                 borderValue=255)
                
                # 快速評估相似度（使用簡化的模板匹配）
                h1, w1 = img1_gray.shape[:2]
                h2_t, w2_t = img2_transformed.shape[:2]
                
                if w2_t <= w1 and h2_t <= h1:
                    try:
                        result = cv2.matchTemplate(img1_gray, img2_transformed, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(result)
                        score = float(max_val)
                        
                        if score > best_score:
                            best_score = score
                            best_offset_x = offset_x
                            best_offset_y = offset_y
                    except Exception:
                        continue
        
        return best_offset_x, best_offset_y, best_score
    
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

