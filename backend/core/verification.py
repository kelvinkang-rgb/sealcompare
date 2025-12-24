"""
校正驗證模組
用於生成校正驗證視覺化和計算驗證指標
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import json
from PIL import Image, ImageDraw, ImageFont


def create_correction_comparison(
    image1_path: str,
    image2_corrected_path: str,
    output_path: Path,
    record_id: int,
    rotation_angle: Optional[float] = None,
    translation_offset: Optional[Dict[str, int]] = None
) -> Optional[str]:
    """
    生成校正對比圖（兩圖並排：只顯示最後比對的兩個圖檔，已裁切與去背）
    
    圖像處理：
    - 保留圖像原始尺寸，不進行尺寸調整（resize）
    - 兩個圖像並排顯示，各自保持原始大小
    - 畫布高度使用兩個圖像中較大的高度
    
    Args:
        image1_path: 圖像1路徑（已裁切和去背景）
        image2_corrected_path: 圖像2路徑（已裁切、去背景和對齊）
        output_path: 輸出目錄
        record_id: 記錄 ID
        rotation_angle: 旋轉角度（度）
        translation_offset: 平移偏移量 {"x": int, "y": int}
        
    Returns:
        生成的對比圖相對路徑，失敗返回 None
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
        
        img1_path = normalize_path(image1_path)
        img2_corr_path = normalize_path(image2_corrected_path)
        
        if not img1_path or not img2_corr_path:
            return None
        
        # 檢查檔案是否存在
        if not img1_path.exists() or not img2_corr_path.exists():
            return None
        
        # 讀取圖像
        img1 = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
        img2_corr = cv2.imread(str(img2_corr_path), cv2.IMREAD_COLOR)
        
        if img1 is None or img2_corr is None:
            return None
        
        # 獲取圖像尺寸（保留原始大小）
        h1, w1 = img1.shape[:2]
        h2_corr, w2_corr = img2_corr.shape[:2]
        
        # 創建並排對比圖（兩圖並排，使用原始尺寸）
        gap = 20  # 圖像間距
        comparison_width = w1 + w2_corr + gap
        comparison_height = max(h1, h2_corr) + 100  # 額外空間用於標註
        
        comparison = np.ones((comparison_height, comparison_width, 3), dtype=np.uint8) * 255
        
        # 放置圖像（使用原始尺寸）
        y_offset = 50  # 頂部留白用於標註
        comparison[y_offset:y_offset+h1, 0:w1] = img1
        comparison[y_offset:y_offset+h2_corr, w1+gap:w1+gap+w2_corr] = img2_corr
        
        # 添加標註文字（使用 PIL 支持中文）
        # 將 OpenCV 圖像轉換為 PIL 圖像
        comparison_pil = Image.fromarray(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(comparison_pil)
        
        # 嘗試載入中文字體，如果失敗則使用默認字體
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',  # Linux
            '/System/Library/Fonts/PingFang.ttc',  # macOS
            'C:/Windows/Fonts/msyh.ttc',  # Windows 微軟雅黑
            'C:/Windows/Fonts/simhei.ttf',  # Windows 黑體
        ]
        font = None
        used_font_path = None
        try:
            for font_path in font_paths:
                if Path(font_path).exists():
                    font = ImageFont.truetype(font_path, 24)
                    used_font_path = font_path
                    break
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # 圖像1標註
        text1 = "參考圖像 (圖像1)"
        bbox = draw.textbbox((0, 0), text1, font=font)
        text_width = bbox[2] - bbox[0]
        text_x1 = (w1 - text_width) // 2
        draw.text((text_x1, 10), text1, fill=(0, 0, 0), font=font)
        
        # 圖像2校正標註
        correction_parts = []
        if rotation_angle is not None and rotation_angle != 0:
            correction_parts.append(f"旋轉 {rotation_angle:.2f}度")
        if translation_offset and (translation_offset.get('x', 0) != 0 or translation_offset.get('y', 0) != 0):
            correction_parts.append(f"平移 ({translation_offset.get('x', 0)}, {translation_offset.get('y', 0)})")
        
        if correction_parts:
            text2 = f"校正後圖像 (圖像2, {', '.join(correction_parts)})"
        else:
            text2 = "校正後圖像 (圖像2)"
        bbox = draw.textbbox((0, 0), text2, font=font)
        text_width = bbox[2] - bbox[0]
        text_x2 = w1 + gap + (w2_corr - text_width) // 2
        
        # 確保文字不會超出圖像範圍
        text_x2 = max(w1 + gap, min(text_x2, comparison_width - text_width - 10))
        # 如果文字仍然太長，可以分成兩行顯示
        if text_width > w2_corr - 20:
            # 將文字分成兩行：主標題和校正信息
            text2_line1 = "校正後圖像 (圖像2)"
            text2_line2 = ", ".join(correction_parts) if correction_parts else ""
            
            bbox1 = draw.textbbox((0, 0), text2_line1, font=font)
            text_width1 = bbox1[2] - bbox1[0]
            text_x2_1 = w1 + gap + (w2_corr - text_width1) // 2
            text_x2_1 = max(w1 + gap, min(text_x2_1, comparison_width - text_width1 - 10))
            draw.text((text_x2_1, 10), text2_line1, fill=(0, 0, 0), font=font)
            
            if text2_line2:
                # 使用稍小的字體顯示校正信息
                font_small = font
                if used_font_path and font != ImageFont.load_default():
                    try:
                        font_small = ImageFont.truetype(used_font_path, 18)
                    except:
                        font_small = font
                bbox2 = draw.textbbox((0, 0), text2_line2, font=font_small)
                text_width2 = bbox2[2] - bbox2[0]
                text_x2_2 = w1 + gap + (w2_corr - text_width2) // 2
                text_x2_2 = max(w1 + gap, min(text_x2_2, comparison_width - text_width2 - 10))
                draw.text((text_x2_2, 35), text2_line2, fill=(0, 0, 0), font=font_small)
        else:
            draw.text((text_x2, 10), text2, fill=(0, 0, 0), font=font)
        
        # 轉回 OpenCV 格式
        comparison = cv2.cvtColor(np.array(comparison_pil), cv2.COLOR_RGB2BGR)
        
        # 添加分隔線
        line_y = y_offset - 10
        cv2.line(comparison, (0, line_y), (comparison_width, line_y), (200, 200, 200), 2)
        
        # 保存對比圖（使用高質量 JPEG，確保解析度與疊圖一致）
        output_path.mkdir(parents=True, exist_ok=True)
        comparison_file = output_path / f"comparison_{record_id}.jpg"
        # 使用 JPEG 質量 95（高質量）以確保解析度一致
        cv2.imwrite(str(comparison_file), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # 返回相對路徑（相對於 logs 目錄）
        return f"comparisons/comparison_{record_id}.jpg"
        
    except Exception as e:
        print(f"警告：無法生成校正對比圖 {record_id}: {e}")
        return None


def create_difference_heatmap(
    image1_path: str,
    image2_corrected_path: Optional[str],
    image2_original_path: str,
    output_path: Path,
    record_id: int
) -> Tuple[Optional[str], Dict]:
    """
    生成差異熱力圖
    
    圖像處理：
    - 如果兩個圖像尺寸不同，會調整到相同尺寸（使用最大尺寸）
    - 使用 INTER_LINEAR 插值以保持高質量
    - 計算像素差異並生成熱力圖（JET 顏色映射：藍→綠→黃→紅）
    - 使用 alpha 混合（0.6）將熱力圖疊加在原圖上
    
    Args:
        image1_path: 圖像1路徑
        image2_corrected_path: 校正後圖像2路徑（優先使用）
        image2_original_path: 原始圖像2路徑（備用）
        output_path: 輸出目錄
        record_id: 記錄 ID
        
    Returns:
        (熱力圖相對路徑, 差異統計字典)，失敗返回 (None, {})
    """
    try:
        # 轉換路徑
        def normalize_path(p):
            if not p:
                return None
            s = str(p)
            if s.startswith('/app/'):
                s = s.replace('/app/', '')
            return Path(s)
        
        img1_path = normalize_path(image1_path)
        img2_path = normalize_path(image2_corrected_path) if image2_corrected_path else normalize_path(image2_original_path)
        
        if not img1_path or not img2_path:
            return None, {}
        
        if not img1_path.exists() or not img2_path.exists():
            return None, {}
        
        # 讀取圖像
        img1 = cv2.imread(str(img1_path), cv2.IMREAD_COLOR)
        img2 = cv2.imread(str(img2_path), cv2.IMREAD_COLOR)
        
        if img1 is None or img2 is None:
            return None, {}
        
        # 調整到相同尺寸（使用高質量插值，與其他視覺化保持一致）
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        target_h = max(h1, h2)
        target_w = max(w1, w2)
        
        # 使用 INTER_LINEAR 插值以保持高質量
        img1_resized = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img2_resized = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # 轉換為灰度圖
        gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        
        # 計算像素差異
        diff = cv2.absdiff(gray1, gray2)
        
        # 應用高斯模糊平滑熱力圖
        diff_blurred = cv2.GaussianBlur(diff, (15, 15), 0)
        
        # 正規化到 0-255
        diff_normalized = cv2.normalize(diff_blurred, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # 應用顏色映射（JET 顏色映射：藍→綠→黃→紅）
        heatmap = cv2.applyColorMap(diff_normalized, cv2.COLORMAP_JET)
        
        # 創建混合圖像（原圖 + 熱力圖）
        # 使用 alpha 混合，讓原圖可見
        alpha = 0.6
        blended = cv2.addWeighted(img1_resized, 1 - alpha, heatmap, alpha, 0)
        
        # 計算差異統計
        total_pixels = target_h * target_w
        diff_pixels = np.count_nonzero(diff > 10)  # 差異閾值為 10
        diff_percentage = (diff_pixels / total_pixels) * 100
        
        # 計算最大差異值
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # 找到最大差異區域（使用滑動窗口）
        window_size = 50
        max_region_diff = 0
        max_region_pos = (0, 0)
        
        for y in range(0, target_h - window_size, window_size // 2):
            for x in range(0, target_w - window_size, window_size // 2):
                region = diff[y:y+window_size, x:x+window_size]
                region_mean = np.mean(region)
                if region_mean > max_region_diff:
                    max_region_diff = region_mean
                    max_region_pos = (x, y)
        
        # 在熱力圖上標註最大差異區域
        cv2.rectangle(blended, max_region_pos, 
                     (max_region_pos[0] + window_size, max_region_pos[1] + window_size),
                     (255, 255, 255), 2)
        cv2.putText(blended, "Max Diff", 
                   (max_region_pos[0], max_region_pos[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 添加圖例和統計資訊
        legend_height = 80
        final_image = np.ones((target_h + legend_height, target_w, 3), dtype=np.uint8) * 255
        final_image[0:target_h, 0:target_w] = blended
        
        # 添加統計文字（使用 PIL 支持中文）
        final_image_pil = Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(final_image_pil)
        
        # 嘗試載入中文字體
        font_paths_heatmap = [
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/System/Library/Fonts/PingFang.ttc',
            'C:/Windows/Fonts/msyh.ttc',
            'C:/Windows/Fonts/simhei.ttf',
        ]
        font_heatmap = None
        used_font_path_heatmap = None
        try:
            for font_path in font_paths_heatmap:
                if Path(font_path).exists():
                    font_heatmap = ImageFont.truetype(font_path, 18)
                    used_font_path_heatmap = font_path
                    break
            if font_heatmap is None:
                font_heatmap = ImageFont.load_default()
        except:
            font_heatmap = ImageFont.load_default()
        
        y_text = target_h + 20
        
        # 只顯示差異像素指標
        stats_text = f"差異像素: {diff_pixels:,} ({diff_percentage:.2f}%)"
        
        # 計算圖例所需空間
        legend_width = 200
        legend_x = target_w - legend_width - 10
        max_text_width = legend_x - 20  # 確保文字不會與圖例重疊
        
        # 檢查文字寬度，如果太長則調整字體
        bbox = draw.textbbox((0, 0), stats_text, font=font_heatmap)
        text_width = bbox[2] - bbox[0]
        if text_width > max_text_width and used_font_path_heatmap:
            # 如果文字太長，使用較小的字體
            try:
                font_small = ImageFont.truetype(used_font_path_heatmap, 14)
                draw.text((10, y_text), stats_text, fill=(0, 0, 0), font=font_small)
            except:
                draw.text((10, y_text), stats_text, fill=(0, 0, 0), font=font_heatmap)
        else:
            draw.text((10, y_text), stats_text, fill=(0, 0, 0), font=font_heatmap)
        
        # 添加顏色圖例
        legend_y = target_h + 10
        legend_h = 20
        
        # 創建顏色條
        color_bar = np.zeros((legend_h, legend_width, 3), dtype=np.uint8)
        for x in range(legend_width):
            ratio = x / legend_width
            color_value = cv2.applyColorMap(np.array([[int(ratio * 255)]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
            color_bar[:, x] = color_value
        
        # 將顏色條轉換為 PIL 並貼上
        color_bar_pil = Image.fromarray(cv2.cvtColor(color_bar, cv2.COLOR_BGR2RGB))
        final_image_pil.paste(color_bar_pil, (legend_x, legend_y))
        
        # 重新創建 draw 對象
        draw = ImageDraw.Draw(final_image_pil)
        
        # 添加圖例標籤
        draw.text((legend_x - 25, legend_y + 5), "低", fill=(0, 0, 0), font=font_heatmap)
        draw.text((legend_x + legend_width + 5, legend_y + 5), "高", fill=(0, 0, 0), font=font_heatmap)
        
        # 轉回 OpenCV 格式
        final_image = cv2.cvtColor(np.array(final_image_pil), cv2.COLOR_RGB2BGR)
        
        # 保存熱力圖（使用高質量 JPEG，確保解析度與其他視覺化一致）
        output_path.mkdir(parents=True, exist_ok=True)
        heatmap_file = output_path / f"heatmap_{record_id}.jpg"
        # 使用 JPEG 質量 95（高質量）以確保解析度一致
        cv2.imwrite(str(heatmap_file), final_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # 差異統計
        stats = {
            'diff_pixels': int(diff_pixels),
            'diff_percentage': round(diff_percentage, 2),
            'mean_diff': round(float(mean_diff), 2),
            'max_diff': int(max_diff),
            'max_region_pos': max_region_pos,
            'max_region_diff': round(float(max_region_diff), 2)
        }
        
        # 返回相對路徑（相對於 logs 目錄）
        return f"heatmaps/heatmap_{record_id}.jpg", stats
        
    except Exception as e:
        print(f"警告：無法生成差異熱力圖 {record_id}: {e}")
        return None, {}


def calculate_alignment_metrics(
    image1: np.ndarray,
    image2_original: np.ndarray,
    image2_corrected: Optional[np.ndarray],
    rotation_angle: Optional[float],
    translation_offset: Optional[Dict[str, int]] = None
) -> Dict:
    """
    計算對齊精度指標
    
    圖像處理：
    - 如果兩個圖像尺寸不同，會調整到相同尺寸（使用最大尺寸）
    - 使用 INTER_LINEAR 插值以保持高質量
    - 使用輪廓檢測找到印章中心點
    - 計算中心點偏移距離
    
    Args:
        image1: 圖像1
        image2_original: 原始圖像2
        image2_corrected: 校正後圖像2（如果存在）
        rotation_angle: 旋轉角度（度）
        translation_offset: 平移偏移量 {"x": int, "y": int}
        
    Returns:
        包含對齊指標的字典（rotation_angle, translation_offset, center_offset, size_ratio, has_correction）
    """
    try:
        metrics = {
            'rotation_angle': round(rotation_angle, 2) if rotation_angle is not None else 0.0,
            'translation_offset': translation_offset if translation_offset else {'x': 0, 'y': 0},
            'center_offset': 0.0,
            'size_ratio': 1.0,
            'has_correction': image2_corrected is not None
        }
        
        # 計算尺寸比例
        h1, w1 = image1.shape[:2]
        h2, w2 = image2_original.shape[:2]
        size_ratio = (w1 * h1) / (w2 * h2) if (w2 * h2) > 0 else 1.0
        metrics['size_ratio'] = round(size_ratio, 4)
        
        # 如果有校正後圖像，計算中心點偏移
        if image2_corrected is not None:
            # 轉換為灰度圖
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) if len(image1.shape) == 3 else image1
            gray2_corr = cv2.cvtColor(image2_corrected, cv2.COLOR_BGR2GRAY) if len(image2_corrected.shape) == 3 else image2_corrected
            
            # 調整到相同尺寸
            h1, w1 = gray1.shape[:2]
            h2, w2 = gray2_corr.shape[:2]
            target_h = max(h1, h2)
            target_w = max(w1, w2)
            
            # 使用 INTER_LINEAR 插值以保持高質量
            gray1_resized = cv2.resize(gray1, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            gray2_resized = cv2.resize(gray2_corr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            # 找到印章中心點（使用輪廓）
            def find_seal_center(img):
                # 二值化
                _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                # 找到輪廓
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # 使用最大輪廓
                    largest_contour = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        return (cx, cy)
                # 如果找不到輪廓，返回圖像中心
                h, w = img.shape[:2]
                return (w // 2, h // 2)
            
            center1 = find_seal_center(gray1_resized)
            center2 = find_seal_center(gray2_resized)
            
            # 計算中心點偏移距離
            offset_x = center1[0] - center2[0]
            offset_y = center1[1] - center2[1]
            offset_distance = np.sqrt(offset_x**2 + offset_y**2)
            
            metrics['center_offset'] = round(float(offset_distance), 2)
            metrics['center1'] = center1
            metrics['center2'] = center2
        else:
            metrics['center_offset'] = 0.0
        
        return metrics
        
    except Exception as e:
        print(f"警告：無法計算對齊指標: {e}")
        return {
            'rotation_angle': round(rotation_angle, 2) if rotation_angle is not None else 0.0,
            'center_offset': 0.0,
            'size_ratio': 1.0,
            'has_correction': False
        }

