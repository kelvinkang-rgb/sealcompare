"""
印鑑比對模組
用於比對兩個印章圖像是否完全一致
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


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
        預處理圖像（轉換為灰度圖、調整大小、二值化）
        
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
        
        # 調整大小到標準尺寸（保持長寬比）
        target_size = 1000
        h, w = gray.shape[:2]
        
        if h == 0 or w == 0:
            raise ValueError("圖像尺寸無效")
        
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if new_w == 0 or new_h == 0:
            raise ValueError("調整後的圖像尺寸無效")
        
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 二值化處理
        _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def find_best_rotation_angle(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        找到圖像2相對於圖像1的最佳旋轉角度（優化算法）
        
        使用粗到細（coarse-to-fine）搜索策略：
        1. 第一階段：每15度搜索一次，找到最佳角度範圍（24次）
        2. 第二階段：在最佳角度±15度範圍內，每2度搜索一次（16次）
        3. 第三階段：在最佳角度±2度範圍內，每0.5度搜索一次（9次）
        
        總共約49次旋轉，而不是360次，大幅提升速度。
        
        Args:
            img1: 參考圖像（已預處理）
            img2: 待旋轉圖像（已預處理）
            
        Returns:
            (最佳角度, 旋轉後的圖像2)
        """
        # 使用更小的圖像進行快速評估（進一步加速搜索）
        scale_factor = 0.3  # 縮小到30%以進一步提升速度
        h, w = img1.shape
        small_h, small_w = int(h * scale_factor), int(w * scale_factor)
        
        # 確保最小尺寸不小於50像素
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
        
        # 第一階段：粗搜索（每20度，減少搜索次數）
        best_angle_coarse = 0.0
        best_score_coarse = 0.0
        
        for angle in range(0, 360, 20):  # 從15度改為20度，減少搜索次數
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img2_rotated = cv2.warpAffine(img2_small, M, (small_w, small_h),
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=255)
            score = self._fast_rotation_match(img1_small, img2_rotated)
            
            if score > best_score_coarse:
                best_score_coarse = score
                best_angle_coarse = angle
        
        # 早期退出：如果第一階段分數已經很高，跳過後續階段
        if best_score_coarse > 0.98:
            best_angle = best_angle_coarse
        else:
            # 第二階段：中等搜索（在最佳角度±20度範圍內，每3度）
            best_angle_medium = best_angle_coarse
            best_score_medium = best_score_coarse
            
            # 確保搜索範圍正確（處理負數和超過360的情況）
            # 標準化最佳角度到 0-360 範圍
            best_angle_normalized = best_angle_coarse % 360
            
            # 在最佳角度±20度範圍內搜索，每3度
            angles_to_search = []
            for offset in range(-20, 21, 3):  # -20 到 +20，每3度
                angle = (best_angle_normalized + offset) % 360
                if angle not in angles_to_search:  # 避免重複
                    angles_to_search.append(angle)
            
            # 限制搜索次數，避免無限循環（最多14次：-20到+20，每3度）
            max_search_count = min(len(angles_to_search), 15)
            for angle in angles_to_search[:max_search_count]:
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img2_rotated = cv2.warpAffine(img2_small, M, (small_w, small_h),
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=255)
                score = self._fast_rotation_match(img1_small, img2_rotated)
                
                if score > best_score_medium:
                    best_score_medium = score
                    best_angle_medium = angle
            
            # 早期退出：如果第二階段分數已經很高，跳過細搜索
            if best_score_medium > 0.98:
                best_angle = best_angle_medium
            else:
                # 第三階段：細搜索（在最佳角度±3度範圍內，每0.5度）
                best_angle = best_angle_medium
                best_score = best_score_medium
                
                # 使用浮點數角度進行細搜索，限制搜索次數
                offsets = np.arange(-3.0, 3.5, 0.5)
                for offset in offsets:
                    angle = (best_angle_medium + offset) % 360
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    img2_rotated = cv2.warpAffine(img2_small, M, (small_w, small_h),
                                                 borderMode=cv2.BORDER_CONSTANT,
                                                 borderValue=255)
                    score = self._fast_rotation_match(img1_small, img2_rotated)
                    
                    if score > best_score:
                        best_score = score
                        best_angle = angle
        
        # 使用最佳角度旋轉原始尺寸的圖像2
        h2, w2 = img2.shape
        center_full = (w2 // 2, h2 // 2)
        M = cv2.getRotationMatrix2D(center_full, best_angle, 1.0)
        img2_rotated_full = cv2.warpAffine(img2, M, (w2, h2),
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=255)
        
        return best_angle, img2_rotated_full
    
    def _fast_rotation_match(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        快速旋轉匹配評估（用於角度搜索）
        
        使用多種快速方法組合，優化計算速度：
        1. 邊緣匹配（Canny + 模板匹配）- 對旋轉最敏感
        2. 像素相似度 - 快速評估
        3. 輪廓匹配 - 僅對小圖像使用
        
        Args:
            img1: 參考圖像
            img2: 待比對圖像
            
        Returns:
            匹配分數 (0-1)
        """
        # 方法1：邊緣檢測 + 模板匹配（最快且對旋轉敏感）
        # 使用自適應閾值以適應不同圖像
        edges1 = cv2.Canny(img1, 50, 150)
        edges2 = cv2.Canny(img2, 50, 150)
        
        edge_match = 0.0
        if np.sum(edges1) > 0 and np.sum(edges2) > 0:
            # 使用模板匹配
            result = cv2.matchTemplate(edges1, edges2, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            edge_match = float(max_val)
        elif np.sum(edges1) == 0 and np.sum(edges2) == 0:
            # 如果都沒有邊緣，可能是空白圖像，使用像素相似度
            edge_match = 1.0
        
        # 方法2：像素相似度（快速，但對旋轉不敏感）
        # 只計算非背景像素的相似度（背景通常是255）
        mask1 = img1 < 250  # 非背景像素
        mask2 = img2 < 250
        mask_combined = mask1 & mask2
        
        if np.sum(mask_combined) > 0:
            diff = cv2.absdiff(img1, img2)
            pixel_similarity = 1.0 - (np.sum(diff[mask_combined] > 10) / np.sum(mask_combined))
        else:
            pixel_similarity = 0.0
        
        # 方法3：輪廓匹配（僅對小圖像使用，計算較慢）
        contour_match = 0.0
        try:
            img_size = img1.shape[0] * img1.shape[1]
            if img_size < 40000:  # 只對小圖像使用（約200x200以下）
                contours1, _ = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours2, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours1) > 0 and len(contours2) > 0:
                    # 使用最大輪廓進行匹配
                    c1 = max(contours1, key=cv2.contourArea)
                    c2 = max(contours2, key=cv2.contourArea)
                    if cv2.contourArea(c1) > 10 and cv2.contourArea(c2) > 10:
                        match = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I2, 0)
                        contour_match = 1.0 / (1.0 + match * 10)  # 轉換為相似度
        except Exception:
            contour_match = 0.0
        
        # 加權組合（邊緣匹配最重要，因為對旋轉敏感）
        if contour_match > 0:
            score = edge_match * 0.6 + pixel_similarity * 0.25 + contour_match * 0.15
        else:
            score = edge_match * 0.7 + pixel_similarity * 0.3
        
        return score
    
    def compare_images(self, image1: np.ndarray, image2: np.ndarray, 
                      enable_rotation_search: bool = True) -> Tuple[bool, float, dict, Optional[np.ndarray]]:
        """
        比對兩個圖像
        
        Args:
            image1: 第一個圖像
            image2: 第二個圖像
            enable_rotation_search: 是否啟用旋轉角度搜索（預設為 True）
            
        Returns:
            (是否一致, 相似度, 詳細資訊, 校正後的圖像2)
        """
        # 輸入驗證
        if image1 is None or image2 is None:
            raise ValueError("圖像不能為 None")
        
        if image1.size == 0 or image2.size == 0:
            raise ValueError("圖像不能為空")
        
        # 輸入驗證
        if image1 is None or image2 is None:
            raise ValueError("圖像不能為 None")
        
        if image1.size == 0 or image2.size == 0:
            raise ValueError("圖像不能為空")
        
        # 預處理圖像
        img1_processed = self.preprocess_image(image1)
        img2_processed_original = self.preprocess_image(image2)
        
        # 計算校正前相似度（用於對比）
        similarity_before_correction = None
        if enable_rotation_search:
            # 確保兩個圖像尺寸相同（校正前）
            h1_orig, w1_orig = img1_processed.shape
            h2_orig, w2_orig = img2_processed_original.shape
            
            if h1_orig != h2_orig or w1_orig != w2_orig:
                target_h_orig = max(h1_orig, h2_orig)
                target_w_orig = max(w1_orig, w2_orig)
                img1_orig_resized = cv2.resize(img1_processed, (target_w_orig, target_h_orig))
                img2_orig_resized = cv2.resize(img2_processed_original, (target_w_orig, target_h_orig))
            else:
                img1_orig_resized = img1_processed
                img2_orig_resized = img2_processed_original
            
            # 計算校正前相似度
            ssim_before = self._calculate_ssim(img1_orig_resized, img2_orig_resized)
            template_before = self._template_match(img1_orig_resized, img2_orig_resized)
            pixel_diff_before = self._pixel_difference(img1_orig_resized, img2_orig_resized)
            similarity_before_correction = (ssim_before * 0.5 + template_before * 0.3 + (1 - pixel_diff_before) * 0.2)
        
        # 旋轉角度搜索（找到圖像2的最佳旋轉角度）
        best_angle = 0.0
        img2_corrected = None
        img2_processed = img2_processed_original.copy()
        if enable_rotation_search:
            best_angle, img2_processed = self.find_best_rotation_angle(img1_processed, img2_processed_original)
            # 保存校正後的原始圖像（用於疊圖比對）
            h2, w2 = image2.shape[:2]
            center = (w2 // 2, h2 // 2)
            M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
            img2_corrected = cv2.warpAffine(image2, M, (w2, h2),
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=(255, 255, 255))
        
        # 確保兩個圖像尺寸相同
        h1, w1 = img1_processed.shape
        h2, w2 = img2_processed.shape
        
        if h1 != h2 or w1 != w2:
            # 調整到相同尺寸
            target_h = max(h1, h2)
            target_w = max(w1, w2)
            img1_processed = cv2.resize(img1_processed, (target_w, target_h))
            img2_processed = cv2.resize(img2_processed, (target_w, target_h))
        
        # 方法1：結構相似性指數 (SSIM)
        similarity_ssim = self._calculate_ssim(img1_processed, img2_processed)
        
        # 方法2：模板匹配
        similarity_template = self._template_match(img1_processed, img2_processed)
        
        # 方法3：像素差異
        pixel_diff = self._pixel_difference(img1_processed, img2_processed)
        
        # 綜合相似度（加權平均）
        similarity = (similarity_ssim * 0.5 + similarity_template * 0.3 + (1 - pixel_diff) * 0.2)
        
        # 判斷是否一致
        is_match = similarity >= self.threshold
        
        # 計算改善幅度
        improvement = None
        if similarity_before_correction is not None:
            improvement = similarity - similarity_before_correction
        
        # 計算尺寸資訊
        h1_orig, w1_orig = image1.shape[:2]
        h2_orig, w2_orig = image2.shape[:2]
        size_ratio = (w1_orig * h1_orig) / (w2_orig * h2_orig) if (w2_orig * h2_orig) > 0 else 1.0
        
        details = {
            'similarity': similarity,
            'ssim': similarity_ssim,
            'template_match': similarity_template,
            'pixel_diff': pixel_diff,
            'threshold': self.threshold,
            'rotation_angle': round(best_angle, 2) if enable_rotation_search else None,
            'similarity_before_correction': round(similarity_before_correction, 4) if similarity_before_correction is not None else None,
            'improvement': round(improvement, 4) if improvement is not None else None,
            'image1_size': (h1_orig, w1_orig),
            'image2_size': (h2_orig, w2_orig),
            'size_ratio': round(size_ratio, 4)
        }
        
        return is_match, similarity, details, img2_corrected
    
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
        模板匹配
        
        Args:
            img1: 第一個圖像
            img2: 第二個圖像
            
        Returns:
            匹配度 (0-1)
        """
        # 使用模板匹配
        result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
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
                     enable_rotation_search: bool = True) -> Tuple[bool, float, dict, Optional[np.ndarray]]:
        """
        比對兩個圖像文件
        
        Args:
            image1_path: 第一個圖像路徑
            image2_path: 第二個圖像路徑
            enable_rotation_search: 是否啟用旋轉角度搜索（預設為 True）
            
        Returns:
            (是否一致, 相似度, 詳細資訊, 校正後的圖像2)
        """
        # 輸入驗證
        if not image1_path or not image2_path:
            raise ValueError("圖像路徑不能為空")
        
        img1 = self.load_image(image1_path)
        img2 = self.load_image(image2_path)
        
        if img1 is None or img2 is None:
            error_msg = f"無法載入圖像: image1={image1_path}, image2={image2_path}"
            return False, 0.0, {'error': error_msg}, None
        
        try:
            return self.compare_images(img1, img2, enable_rotation_search=enable_rotation_search)
        except Exception as e:
            error_msg = f"比對過程中發生錯誤: {str(e)}"
            return False, 0.0, {'error': error_msg}, None

