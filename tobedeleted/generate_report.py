"""
生成比對測試報告
從記錄檔案中讀取所有比對結果並生成報告
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
from verification import (
    create_correction_comparison,
    create_difference_heatmap,
    calculate_alignment_metrics
)


def create_overlay_image(image1_path: str, image2_path: str, overlay_dir: Path, record_id: int, 
                         image2_corrected_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
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
        
        # 調整到相同尺寸
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        target_h = max(h1, h2)
        target_w = max(w1, w2)
        
        img1_resized = cv2.resize(img1, (target_w, target_h))
        img2_resized = cv2.resize(img2, (target_w, target_h))
        
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
            # 取邊緣區域的樣本
            edge_width = max(5, min(h, w) // 20)  # 邊緣寬度
            
            # 收集邊緣像素
            edge_pixels = []
            # 上邊緣
            edge_pixels.extend(img[0:edge_width, :].reshape(-1, 3).tolist())
            # 下邊緣
            edge_pixels.extend(img[h-edge_width:h, :].reshape(-1, 3).tolist())
            # 左邊緣
            edge_pixels.extend(img[:, 0:edge_width].reshape(-1, 3).tolist())
            # 右邊緣
            edge_pixels.extend(img[:, w-edge_width:w].reshape(-1, 3).tolist())
            
            edge_colors = np.array(edge_pixels, dtype=np.float32)
            
            # 計算邊緣的主要顏色（使用中位數，對異常值更穩健）
            bg_color = np.median(edge_colors, axis=0).astype(np.uint8)
            
            # 計算每個像素與背景顏色的距離
            img_float = img.astype(np.float32)
            bg_float = bg_color.astype(np.float32)
            
            # 計算顏色距離（歐氏距離）
            color_diff = np.sqrt(np.sum((img_float - bg_float) ** 2, axis=2))
            
            # 設定閾值：如果顏色距離小於閾值，則認為是背景
            # 對於印章圖像，背景通常是單色或接近單色
            # 使用自適應閾值：根據邊緣顏色的標準差調整
            edge_std = np.std(edge_colors, axis=0).mean()
            threshold = max(25, min(50, edge_std * 2))  # 動態調整閾值
            
            # 創建背景遮罩（背景為 True）
            bg_mask = color_diff < threshold
            
            # 方法2: 如果圖像是高亮度的（可能是白色背景），使用亮度閾值
            mean_brightness = np.mean(gray)
            
            if mean_brightness > 200:  # 圖像很亮，可能是白色背景
                # 使用高閾值識別白色背景
                _, bright_bg_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
                bright_bg_mask = bright_bg_mask > 0
                
                # 合併兩種方法：取並集（更保守，確保背景被正確識別）
                bg_mask = bg_mask | bright_bg_mask
            elif mean_brightness < 50:  # 圖像很暗，可能是黑色背景
                # 使用低閾值識別黑色背景
                _, dark_bg_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
                dark_bg_mask = dark_bg_mask > 0
                bg_mask = bg_mask | dark_bg_mask
            
            # 印章遮罩是背景的反轉
            seal_mask = (~bg_mask).astype(np.uint8) * 255
            
            # 形態學操作優化遮罩
            kernel = np.ones((3, 3), np.uint8)
            # 閉運算：填充印章內部的小洞
            seal_mask = cv2.morphologyEx(seal_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            # 開運算：去除小的噪點
            seal_mask = cv2.morphologyEx(seal_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 如果印章區域太小，可能是檢測錯誤，使用 OTSU 作為備選
            seal_area = np.sum(seal_mask > 0)
            total_area = h * w
            if seal_area < total_area * 0.01:  # 印章區域小於 1%
                # 使用 OTSU 自動閾值
                _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                seal_mask = binary_otsu
                # 形態學操作
                seal_mask = cv2.morphologyEx(seal_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                seal_mask = cv2.morphologyEx(seal_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 確保印章 mask 是二值化的（0 或 255）
            _, seal_mask = cv2.threshold(seal_mask, 127, 255, cv2.THRESH_BINARY)
            
            # 創建 RGBA 圖像
            rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            
            # 將背景設為透明（alpha = 0）
            # seal_mask 中，印章部分為 255，背景為 0
            rgba[:, :, 3] = seal_mask
            
            return seal_mask, rgba
        
        # 處理兩個圖像
        mask1, img1_rgba = remove_background_and_make_transparent(img1_resized)
        mask2, img2_rgba = remove_background_and_make_transparent(img2_resized)
        
        # 轉換為灰度圖以便比對（使用 mask）
        gray1 = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2GRAY) if len(img1_resized.shape) == 3 else img1_resized
        gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY) if len(img2_resized.shape) == 3 else img2_resized
        
        # 使用 mask 來獲取印章部分的二值化圖像
        # 只考慮印章部分（mask > 0 的區域）
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
        overlap_mask = (binary1 > 0) & (binary2 > 0)  # 兩者都有（重疊）
        
        # 疊圖1：圖像1（藍色）疊在圖像2（紅色）上，顯示圖像2多出的部分（黃色）
        # 先畫圖像2的基礎（紅色）
        overlay1_on_2[binary2 > 0] = [0, 0, 255]  # 紅色（圖像2，BGR格式）
        # 再畫圖像1（藍色，會覆蓋重疊部分）
        overlay1_on_2[binary1 > 0] = [255, 0, 0]  # 藍色（圖像1，BGR格式）
        # 最後標示圖像2多出的部分（黃色，會覆蓋藍色）
        overlay1_on_2[diff_mask_2_only] = [0, 255, 255]  # 黃色（圖像2多出部分，BGR格式）
        
        # 疊圖2：圖像2（紅色）疊在圖像1（藍色）上，顯示圖像1多出的部分（黃色）
        # 先畫圖像1的基礎（藍色）
        overlay2_on_1[binary1 > 0] = [255, 0, 0]  # 藍色（圖像1，BGR格式）
        # 再畫圖像2（紅色，會覆蓋重疊部分）
        overlay2_on_1[binary2 > 0] = [0, 0, 255]  # 紅色（圖像2，BGR格式）
        # 最後標示圖像1多出的部分（黃色，會覆蓋紅色）
        overlay2_on_1[diff_mask_1_only] = [0, 255, 255]  # 黃色（圖像1多出部分，BGR格式）
        
        # 創建帶透明背景的疊圖（使用 PNG 格式支持透明度）
        # 將疊圖轉換為 RGBA 格式
        overlay1_rgba = np.zeros((target_h, target_w, 4), dtype=np.uint8)
        overlay2_rgba = np.zeros((target_h, target_w, 4), dtype=np.uint8)
        
        # 疊圖1：圖像1（藍色）疊在圖像2（紅色）上
        overlay1_rgba[:, :, :3] = overlay1_on_2  # BGR 通道
        # Alpha 通道：只有印章部分不透明
        overlay1_rgba[:, :, 3] = np.maximum(binary1, binary2)  # 兩個印章的合併區域
        
        # 疊圖2：圖像2（紅色）疊在圖像1（藍色）上
        overlay2_rgba[:, :, :3] = overlay2_on_1  # BGR 通道
        # Alpha 通道：只有印章部分不透明
        overlay2_rgba[:, :, 3] = np.maximum(binary1, binary2)  # 兩個印章的合併區域
        
        # 保存疊圖（使用 PNG 格式以支持透明度）
        overlay_dir.mkdir(exist_ok=True)
        overlay1_file = overlay_dir / f"overlay_{record_id}_img1_on_img2.png"
        overlay2_file = overlay_dir / f"overlay_{record_id}_img2_on_img1.png"
        
        cv2.imwrite(str(overlay1_file), overlay1_rgba)
        cv2.imwrite(str(overlay2_file), overlay2_rgba)
        
        # 返回相對路徑（相對於 logs 目錄）
        overlay1_url = f"overlays/overlay_{record_id}_img1_on_img2.png"
        overlay2_url = f"overlays/overlay_{record_id}_img2_on_img1.png"
        
        return overlay1_url, overlay2_url
        
    except Exception as e:
        print(f"警告：無法生成疊圖 {record_id}: {e}")
        return None, None


def _generate_verification_html(
    comparison_url: Optional[str],
    heatmap_url: Optional[str],
    heatmap_stats: Dict,
    alignment_metrics: Dict,
    details: Dict,
    similarity: float
) -> str:
    """
    生成校正驗證 HTML
    
    Args:
        comparison_url: 並排對比圖 URL
        heatmap_url: 熱力圖 URL
        heatmap_stats: 熱力圖統計
        alignment_metrics: 對齊指標
        details: 詳細資訊
        similarity: 相似度（已經是百分比，0-100）
        
    Returns:
        HTML 字串
    """
    html_parts = []
    
    # 校正指標卡片
    rotation_angle = alignment_metrics.get('rotation_angle', 0) or details.get('rotation_angle', 0) or 0
    center_offset = alignment_metrics.get('center_offset', 0) or 0
    similarity_before = details.get('similarity_before_correction')
    improvement = details.get('improvement')
    
    metrics_html = f"""
        <div class="verification-metrics">
            <div class="metric-item">
                <span class="metric-label">旋轉角度:</span>
                <span class="metric-value">{rotation_angle:.2f}°</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">中心偏移:</span>
                <span class="metric-value">{center_offset:.2f}px</span>
            </div>
    """
    
    if similarity_before is not None:
        metrics_html += f"""
            <div class="metric-item">
                <span class="metric-label">校正前:</span>
                <span class="metric-value">{similarity_before*100:.2f}%</span>
            </div>
        """
    
    metrics_html += f"""
            <div class="metric-item">
                <span class="metric-label">校正後:</span>
                <span class="metric-value">{similarity:.2f}%</span>
            </div>
    """
    
    if improvement is not None:
        improvement_pct = improvement * 100
        improvement_class = "positive" if improvement_pct > 0 else "negative"
        metrics_html += f"""
            <div class="metric-item">
                <span class="metric-label">改善:</span>
                <span class="metric-value {improvement_class}">{improvement_pct:+.2f}%</span>
            </div>
        """
    
    metrics_html += "</div>"
    html_parts.append(metrics_html)
    
    # 並排對比圖和差異熱力圖容器
    images_html = ""
    if comparison_url or heatmap_url:
        images_html = '<div class="verification-images-container">'
        
        # 並排對比圖
        if comparison_url:
            images_html += f"""
                <div class="verification-image">
                    <a href="javascript:void(0)" onclick="openOverlayModal('{comparison_url}', '校正前後對比圖')" title="點擊查看大圖">
                        <img src="{comparison_url}" alt="校正對比" class="verification-thumbnail" onerror="this.onerror=null; this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'150\' height=\'100\'%3E%3Crect fill=\'%23ddd\' width=\'150\' height=\'100\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' dy=\'.3em\' fill=\'%23999\' font-size=\'10\'%3E對比圖載入失敗%3C/text%3E%3C/svg%3E';">
                    </a>
                    <div class="verification-label">並排對比</div>
                </div>
            """
        
        # 差異熱力圖
        if heatmap_url:
            diff_pct = heatmap_stats.get('diff_percentage', 0)
            diff_pixels = heatmap_stats.get('diff_pixels', 0)
            images_html += f"""
                <div class="verification-image">
                    <a href="javascript:void(0)" onclick="openOverlayModal('{heatmap_url}', '差異熱力圖<br>差異像素: {diff_pixels:,} ({diff_pct:.2f}%)')" title="點擊查看大圖">
                        <img src="{heatmap_url}" alt="差異熱力圖" class="verification-thumbnail" onerror="this.onerror=null; this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'150\' height=\'100\'%3E%3Crect fill=\'%23ddd\' width=\'150\' height=\'100\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' dy=\'.3em\' fill=\'%23999\' font-size=\'10\'%3E熱力圖載入失敗%3C/text%3E%3C/svg%3E';">
                    </a>
                    <div class="verification-label">差異熱力圖</div>
                    <div class="verification-stats">差異: {diff_pct:.2f}%</div>
                </div>
            """
        
        images_html += '</div>'
        html_parts.append(images_html)
    
    if not html_parts:
        return '<div style="color:#999;font-size:12px;">無驗證資料</div>'
    
    return '<div class="verification-section">' + ''.join(html_parts) + '</div>'


def _generate_image2_corrected_html(image2_corrected_path: Optional[str], image2_corrected_url: Optional[str], image2_corrected: Optional[str]) -> str:
    """
    生成圖像2校正的 HTML
    
    Args:
        image2_corrected_path: 校正後圖像路徑
        image2_corrected_url: 校正後圖像 URL
        image2_corrected: 校正後圖像檔名
        
    Returns:
        HTML 字串
    """
    if image2_corrected_path and image2_corrected_url:
        error_svg = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='105' height='105'%3E%3Crect fill='%23ddd' width='105' height='105'/%3E%3Ctext x='50%25' y='50%25' text-anchor='middle' dy='.3em' fill='%23999' font-size='11'%3E圖片載入失敗%3C/text%3E%3C/svg%3E"
        return f"""
                        <div class="image-cell">
                            <a href="{image2_corrected_url}" target="_blank" title="點擊查看原圖">
                                <img src="{image2_corrected_url}" alt="{image2_corrected}" class="thumbnail" onerror="this.onerror=null; this.src='{error_svg}';">
                            </a>
                            <span class="filename">{image2_corrected}</span>
                            <span class="path-text">{image2_corrected_url}</span>
                        </div>
                        """
    else:
        return '<div style="color:#999;font-size:12px;">無校正圖像</div>'


def load_comparison_logs(log_file: Path) -> List[Dict]:
    """
    載入比對記錄
    
    Args:
        log_file: 記錄檔案路徑
        
    Returns:
        記錄列表
    """
    if not log_file.exists():
        print(f"錯誤：找不到記錄檔案 {log_file}")
        return []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
        return records
    except (json.JSONDecodeError, IOError) as e:
        print(f"錯誤：無法讀取記錄檔案: {e}")
        return []


def generate_text_report(records: List[Dict], output_file: Path = None):
    """
    生成文字報告
    
    Args:
        records: 比對記錄列表
        output_file: 輸出檔案路徑（可選）
    """
    if not records:
        print("沒有比對記錄可生成報告")
        return
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("印鑑比對測試報告")
    report_lines.append("=" * 80)
    report_lines.append(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"總測試次數: {len(records)}")
    report_lines.append("")
    
    # 統計資訊
    match_count = sum(1 for r in records if r.get('is_match', False))
    mismatch_count = len(records) - match_count
    avg_similarity = sum(r.get('similarity', 0) for r in records) / len(records) if records else 0
    
    report_lines.append("統計摘要")
    report_lines.append("-" * 80)
    report_lines.append(f"  匹配次數: {match_count} ({match_count/len(records)*100:.1f}%)")
    report_lines.append(f"  不匹配次數: {mismatch_count} ({mismatch_count/len(records)*100:.1f}%)")
    report_lines.append(f"  平均相似度: {avg_similarity*100:.2f}%")
    report_lines.append("")
    
    # 詳細記錄
    report_lines.append("詳細測試記錄")
    report_lines.append("-" * 80)
    
    for i, record in enumerate(records, 1):
        timestamp = record.get('timestamp', 'N/A')
        try:
            dt = datetime.fromisoformat(timestamp)
            timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            timestamp_str = timestamp
        
        image1 = Path(record.get('image1', 'N/A')).name
        image2 = Path(record.get('image2', 'N/A')).name
        is_match = record.get('is_match', False)
        similarity = record.get('similarity', 0) * 100
        threshold = record.get('threshold', 0) * 100
        details = record.get('details', {})
        
        report_lines.append(f"\n測試 #{i}")
        report_lines.append(f"  時間: {timestamp_str}")
        report_lines.append(f"  圖像1: {image1}")
        report_lines.append(f"  圖像2: {image2}")
        report_lines.append(f"  結果: {'✓ 匹配' if is_match else '✗ 不匹配'}")
        report_lines.append(f"  相似度: {similarity:.2f}%")
        report_lines.append(f"  閾值: {threshold:.2f}%")
        
        if details:
            report_lines.append(f"  詳細指標:")
            report_lines.append(f"    - SSIM: {details.get('ssim', 0)*100:.2f}%")
            report_lines.append(f"    - 模板匹配: {details.get('template_match', 0)*100:.2f}%")
            report_lines.append(f"    - 像素差異: {details.get('pixel_diff', 0)*100:.2f}%")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # 輸出報告
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✓ 報告已生成: {output_file}")
        except IOError as e:
            print(f"錯誤：無法寫入報告檔案: {e}")
            print("\n" + report_text)
    else:
        print("\n" + report_text)


def generate_html_report(records: List[Dict], output_file: Path):
    """
    生成 HTML 報告
    
    Args:
        records: 比對記錄列表
        output_file: 輸出檔案路徑
    """
    if not records:
        print("沒有比對記錄可生成報告")
        return
    
    # 按時間戳排序，最新的在最上面
    def get_timestamp(record):
        try:
            ts = record.get('timestamp', '')
            if ts:
                return datetime.fromisoformat(ts).timestamp()
            return 0
        except:
            return 0
    
    records = sorted(records, key=get_timestamp, reverse=True)
    
    # 統計資訊
    match_count = sum(1 for r in records if r.get('is_match', False))
    mismatch_count = len(records) - match_count
    avg_similarity = sum(r.get('similarity', 0) for r in records) / len(records) if records else 0
    
    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>印鑑比對測試報告</title>
    <style>
        body {{
            font-family: 'Microsoft JhengHei', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card.match {{
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        }}
        .stat-card.mismatch {{
            background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
        }}
        .stat-card.avg {{
            background: linear-gradient(135deg, #2196F3 0%, #0b7dda 100%);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .table-wrapper {{
            overflow-x: visible;
            overflow-y: visible;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 100%;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: auto;
        }}
        th, td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
            vertical-align: top;
            word-wrap: break-word;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            white-space: nowrap;
            font-size: 0.9em;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        /* 讓表格行可以換行顯示寬列內容 */
        .row-content-wrapper {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .row-section {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: flex-start;
        }}
        .row-section-compact {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .match-badge {{
            background-color: #4CAF50;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        .mismatch-badge {{
            background-color: #f44336;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        .similarity-bar {{
            background-color: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            position: relative;
            overflow: hidden;
        }}
        .similarity-fill {{
            background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }}
        .details {{
            font-size: 0.85em;
            color: #666;
            line-height: 1.8;
        }}
        .detail-item {{
            margin-bottom: 8px;
        }}
        .detail-label {{
            display: flex;
            align-items: center;
            gap: 4px;
            font-weight: 600;
            margin-bottom: 4px;
            font-size: 0.9em;
        }}
        .detail-icon {{
            font-size: 1em;
        }}
        .detail-value-container {{
            display: flex;
            align-items: center;
            gap: 8px;
            position: relative;
            background: #f5f5f5;
            border-radius: 4px;
            padding: 2px;
            height: 20px;
        }}
        .detail-bar {{
            height: 16px;
            border-radius: 8px;
            min-width: 2px;
            transition: width 0.3s ease;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }}
        .detail-value {{
            font-weight: bold;
            font-size: 0.9em;
            min-width: 50px;
            text-align: right;
            color: #333;
        }}
        td:nth-child(1) {{
            width: 40px;
            text-align: center;
        }}
        td:nth-child(2) {{
            width: 100px;
            white-space: normal;
            line-height: 1.4;
            font-size: 0.85em;
        }}
        td:nth-child(3), td:nth-child(4), td:nth-child(5) {{
            width: 120px;
            max-width: 120px;
        }}
        td:nth-child(6) {{
            width: 70px;
            text-align: center;
        }}
        td:nth-child(7) {{
            width: 90px;
        }}
        td:nth-child(8) {{
            width: 70px;
            text-align: center;
        }}
        td:nth-child(9) {{
            width: 140px;
            max-width: 200px;
        }}
        /* 驗證和疊圖列使用彈性寬度，允許換行 */
        td:nth-child(10), td:nth-child(11) {{
            width: auto;
            min-width: 300px;
            max-width: 100%;
        }}
        .image-cell {{
            text-align: center;
            width: 100%;
            max-width: 120px;
            padding: 2px;
        }}
        .image-cell a {{
            text-decoration: none;
            display: inline-block;
            max-width: 100%;
        }}
        .thumbnail {{
            width: 100%;
            max-width: 80px;
            height: auto;
            max-height: 80px;
            object-fit: contain;
            display: block;
            margin: 0 auto;
        }}
        .filename {{
            display: block;
            margin-top: 4px;
            font-size: 11px;
            color: #555;
            word-break: break-word;
            line-height: 1.3;
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .path-text {{
            display: block;
            font-size: 9px;
            color: #777;
            word-break: break-all;
            line-height: 1.2;
            margin-top: 1px;
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .overlay-section {{
            margin-top: 2px;
            padding: 4px;
            background: #f9fafb;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
            width: 100%;
            max-width: 100%;
        }}
        .overlay-images {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .overlay-item {{
            text-align: center;
            padding: 0;
            margin: 0;
        }}
        .overlay-item a {{
            display: inline-block;
            max-width: 100%;
        }}
        .overlay-item img.overlay-thumbnail {{
            width: 100%;
            max-width: 180px;
            height: auto;
            max-height: 120px;
            object-fit: contain;
            border: 2px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .overlay-item img.overlay-thumbnail:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .overlay-item a {{
            text-decoration: none;
        }}
        /* 驗證區塊樣式 */
        .verification-section {{
            margin-top: 2px;
            padding: 4px;
            background: #f0f9ff;
            border-radius: 8px;
            border: 1px solid #bae6fd;
            width: 100%;
            max-width: 100%;
        }}
        .verification-metrics {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 3px;
            margin-bottom: 6px;
            padding: 4px;
            background: white;
            border-radius: 4px;
            font-size: 10px;
        }}
        .metric-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 2px 4px;
        }}
        .metric-label {{
            color: #666;
            font-weight: normal;
        }}
        .metric-value {{
            color: #333;
            font-weight: bold;
        }}
        .metric-value.positive {{
            color: #059669;
        }}
        .metric-value.negative {{
            color: #dc2626;
        }}
        .verification-images-container {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .verification-image {{
            text-align: center;
        }}
        .verification-image a {{
            display: inline-block;
            max-width: 100%;
        }}
        .verification-thumbnail {{
            width: 100%;
            max-width: 200px;
            height: auto;
            max-height: 100px;
            object-fit: contain;
            border: 2px solid #93c5fd;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            background: white;
        }}
        .verification-thumbnail:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            z-index: 10;
            position: relative;
        }}
        .verification-label {{
            font-size: 8px;
            color: #1e40af;
            margin-top: 2px;
            font-weight: bold;
        }}
        .verification-stats {{
            font-size: 8px;
            color: #666;
            margin-top: 1px;
        }}
        /* 模態框樣式 */
        .overlay-modal {{
            display: none;
            position: fixed;
            z-index: 10000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
            overflow: auto;
        }}
        .overlay-modal-content {{
            position: relative;
            margin: 2% auto;
            padding: 20px;
            width: 90%;
            max-width: 1200px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }}
        .overlay-modal-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }}
        .overlay-modal-title {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }}
        .overlay-modal-close {{
            color: #aaa;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            line-height: 20px;
        }}
        .overlay-modal-close:hover {{
            color: #000;
        }}
        .overlay-modal-image-container {{
            text-align: center;
            background: #f5f5f5;
            padding: 20px;
            border-radius: 4px;
        }}
        .overlay-modal-image {{
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .overlay-modal-description {{
            margin-top: 15px;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 4px;
            text-align: center;
            color: #666;
            font-size: 14px;
        }}
            transition: transform 0.2s;
            background: #fff;
            display: block;
            margin: 0 auto;
        }}
        .overlay-item img:hover {{
            transform: scale(1.3);
            z-index: 10;
            position: relative;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }}
        .overlay-label {{
            font-size: 9px;
            color: #6b7280;
            margin-top: 2px;
            line-height: 1.2;
            word-break: break-word;
            padding: 0;
        }}
        .thumbnail {{
            width: 100%;
            max-width: 80px;
            height: auto;
            max-height: 80px;
            object-fit: contain;
            border: 2px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s;
            background: #fff;
            display: block;
            margin: 0 auto;
        }}
        .thumbnail:hover {{
            transform: scale(1.5);
            z-index: 10;
            position: relative;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <!-- 疊圖模態框 -->
    <div id="overlayModal" class="overlay-modal">
        <div class="overlay-modal-content">
            <div class="overlay-modal-header">
                <div class="overlay-modal-title">疊圖比對詳情</div>
                <span class="overlay-modal-close" onclick="closeOverlayModal()">&times;</span>
            </div>
            <div class="overlay-modal-image-container">
                <img id="overlayModalImage" class="overlay-modal-image" src="" alt="疊圖">
                <div id="overlayModalDescription" class="overlay-modal-description"></div>
            </div>
        </div>
    </div>
    
    <script>
        // 打開疊圖模態框
        function openOverlayModal(imageUrl, description) {{
            const modal = document.getElementById('overlayModal');
            const modalImage = document.getElementById('overlayModalImage');
            const modalDescription = document.getElementById('overlayModalDescription');
            
            // 處理相對路徑（report.html 位於 logs/ 目錄，疊圖也在 logs/overlays/）
            let fullUrl = imageUrl;
            if (!imageUrl.startsWith('http://') && !imageUrl.startsWith('https://') && !imageUrl.startsWith('/')) {{
                // 如果路徑已經是 overlays/ 開頭，則直接使用（因為 report.html 在 logs/ 目錄）
                if (imageUrl.startsWith('overlays/')) {{
                    fullUrl = imageUrl;  // 相對路徑，從 logs/ 目錄開始
                }} else if (!imageUrl.startsWith('../')) {{
                    fullUrl = imageUrl;  // 保持原樣
                }} else {{
                    fullUrl = imageUrl;
                }}
            }}
            
            modalImage.src = fullUrl;
            modalDescription.innerHTML = description;
            modal.style.display = 'block';
            
            // 點擊背景關閉模態框
            modal.onclick = function(event) {{
                if (event.target === modal) {{
                    closeOverlayModal();
                }}
            }};
        }}
        
        // 關閉疊圖模態框
        function closeOverlayModal() {{
            document.getElementById('overlayModal').style.display = 'none';
        }}
        
        // ESC 鍵關閉模態框
        document.addEventListener('keydown', function(event) {{
            if (event.key === 'Escape') {{
                closeOverlayModal();
            }}
        }});
    </script>
    <div class="container">
        <h1>印鑑比對測試報告</h1>
        <p><strong>生成時間:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>總測試次數:</strong> {len(records)}</p>
        
        <div class="summary">
            <div class="stat-card match">
                <div class="stat-label">匹配次數</div>
                <div class="stat-value">{match_count}</div>
                <div class="stat-label">({match_count/len(records)*100:.1f}%)</div>
            </div>
            <div class="stat-card mismatch">
                <div class="stat-label">不匹配次數</div>
                <div class="stat-value">{mismatch_count}</div>
                <div class="stat-label">({mismatch_count/len(records)*100:.1f}%)</div>
            </div>
            <div class="stat-card avg">
                <div class="stat-label">平均相似度</div>
                <div class="stat-value">{avg_similarity*100:.2f}%</div>
            </div>
        </div>
        
        <h2>詳細測試記錄</h2>
        <div class="table-wrapper">
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>時間</th>
                    <th style="display:none;">圖像1</th>
                    <th style="display:none;">圖像2</th>
                    <th style="display:none;">圖像2校正</th>
                    <th>結果</th>
                    <th>相似度</th>
                    <th>閾值</th>
                    <th>詳細指標</th>
                    <th>校正驗證</th>
                    <th>疊圖比對</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # 創建疊圖目錄
    overlay_dir = Path('logs/overlays')
    overlay_dir.mkdir(exist_ok=True)
    
    for i, record in enumerate(records, 1):
        timestamp = record.get('timestamp', 'N/A')
        try:
            dt = datetime.fromisoformat(timestamp)
            date_str = dt.strftime('%Y-%m-%d')
            time_str = dt.strftime('%H:%M:%S')
            timestamp_str = f'{date_str}<br>{time_str}'
        except:
            # 如果無法解析，嘗試簡單分割
            if ' ' in timestamp:
                parts = timestamp.split(' ', 1)
                timestamp_str = f'{parts[0]}<br>{parts[1]}'
            else:
                timestamp_str = timestamp
        
        image1_path = record.get('image1', 'N/A')
        image2_path = record.get('image2', 'N/A')
        image2_corrected_path = record.get('image2_corrected', None)
        image1 = Path(image1_path).name
        image2 = Path(image2_path).name
        image2_corrected = Path(image2_corrected_path).name if image2_corrected_path else None
        is_match = record.get('is_match', False)
        similarity = record.get('similarity', 0) * 100
        threshold = record.get('threshold', 0) * 100
        details = record.get('details', {})
        
        # 轉換容器內路徑為相對路徑（用於顯示圖片）
        def get_image_url(image_path):
            """將路徑轉為 report.html 可用的相對路徑（report 位於 logs/）"""
            if not image_path:
                return ''
            path = Path(image_path)
            # 1) 容器內 /app/ 開頭 -> 去除 /app/
            if str(path).startswith('/app/'):
                path = Path(str(path).replace('/app/', ''))
            # 2) Windows 磁碟開頭 -> 去掉磁碟與前導分隔
            path_str = str(path).replace('\\', '/')
            path_str = Path(path_str).as_posix()
            if len(path_str) > 1 and path_str[1:3] == ':/':
                path_str = path_str[3:]
            # 3) 如果已經是 http 或 / 開頭，直接返回
            if path_str.startswith('http://') or path_str.startswith('https://') or path_str.startswith('/'):
                return path_str
            # 4) 若已經有 ../ 則保持，否則加上 ../（因 report.html 位於 logs/）
            if not path_str.startswith('../'):
                path_str = '../' + path_str
            return path_str
        
        image1_url = get_image_url(image1_path)
        image2_url = get_image_url(image2_path)
        image2_corrected_url = get_image_url(image2_corrected_path) if image2_corrected_path else None
        
        # 生成驗證視覺化
        comparison_dir = Path('logs/comparisons')
        heatmap_dir = Path('logs/heatmaps')
        comparison_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        
        rotation_angle = details.get('rotation_angle')
        comparison_url = create_correction_comparison(
            image1_path, image2_path, image2_corrected_path,
            comparison_dir, i, rotation_angle
        )
        
        heatmap_url, heatmap_stats = create_difference_heatmap(
            image1_path, image2_corrected_path, image2_path,
            heatmap_dir, i
        )
        
        # 計算對齊指標（需要讀取圖像）
        alignment_metrics = {}
        try:
            # 處理路徑（去除 /app/ 前綴等）
            def normalize_path_for_read(p):
                if not p:
                    return None
                s = str(p)
                if s.startswith('/app/'):
                    s = s.replace('/app/', '')
                path = Path(s)
                if path.exists():
                    return path
                # 嘗試絕對路徑
                abs_path = Path(p).resolve()
                if abs_path.exists():
                    return abs_path
                return None
            
            img1_path_obj = normalize_path_for_read(image1_path)
            img2_path_obj = normalize_path_for_read(image2_path)
            
            if img1_path_obj and img2_path_obj:
                img1 = cv2.imread(str(img1_path_obj))
                img2_orig = cv2.imread(str(img2_path_obj))
                img2_corr = None
                if image2_corrected_path:
                    img2_corr_path_obj = normalize_path_for_read(image2_corrected_path)
                    if img2_corr_path_obj:
                        img2_corr = cv2.imread(str(img2_corr_path_obj))
                
                if img1 is not None and img2_orig is not None:
                    alignment_metrics = calculate_alignment_metrics(
                        img1, img2_orig, img2_corr, rotation_angle
                    )
        except Exception as e:
            print(f"警告：無法計算對齊指標 {i}: {e}")
        
        # 生成驗證 HTML
        verification_html = _generate_verification_html(
            comparison_url, heatmap_url, heatmap_stats,
            alignment_metrics, details, similarity
        )
        
        # 生成疊圖（使用校正後的圖像2）
        overlay_image2_path = image2_corrected_path if image2_corrected_path else image2_path
        overlay1_url, overlay2_url = create_overlay_image(
            image1_path, image2_path, overlay_dir, i, image2_corrected_path=overlay_image2_path
        )
        
        # 疊圖 HTML（使用 JavaScript 模態框）
        overlay_html = ""
        if overlay1_url and overlay2_url:
            overlay_html = f"""
                <div class="overlay-section">
                    <div class="overlay-images">
                        <div class="overlay-item">
                            <a href="javascript:void(0)" onclick="openOverlayModal('{overlay1_url}', '圖像1(藍)疊在圖像2校正(紅)上<br>黃色=圖像2多出部分')" title="點擊查看大圖">
                                <img src="{overlay1_url}" alt="圖像1疊在圖像2校正上" class="overlay-thumbnail" onerror="this.onerror=null; this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'98\' height=\'98\'%3E%3Crect fill=\'%23ddd\' width=\'98\' height=\'98\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' dy=\'.3em\' fill=\'%23999\' font-size=\'10\'%3E疊圖載入失敗%3C/text%3E%3C/svg%3E';">
                            </a>
                            <div class="overlay-label">圖像1(藍)疊在圖像2校正(紅)上<br>黃色=圖像2多出部分<br><span style="color:#666;font-size:10px;">點擊查看大圖</span></div>
                        </div>
                        <div class="overlay-item">
                            <a href="javascript:void(0)" onclick="openOverlayModal('{overlay2_url}', '圖像2校正(紅)疊在圖像1(藍)上<br>黃色=圖像1多出部分')" title="點擊查看大圖">
                                <img src="{overlay2_url}" alt="圖像2校正疊在圖像1上" class="overlay-thumbnail" onerror="this.onerror=null; this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'98\' height=\'98\'%3E%3Crect fill=\'%23ddd\' width=\'98\' height=\'98\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' dy=\'.3em\' fill=\'%23999\' font-size=\'10\'%3E疊圖載入失敗%3C/text%3E%3C/svg%3E';">
                            </a>
                            <div class="overlay-label">圖像2校正(紅)疊在圖像1(藍)上<br>黃色=圖像1多出部分<br><span style="color:#666;font-size:10px;">點擊查看大圖</span></div>
                        </div>
                    </div>
                </div>
            """
        else:
            overlay_html = '<div class="overlay-section"><div style="color:#999;font-size:12px;">無法生成疊圖</div></div>'
        
        badge_class = 'match-badge' if is_match else 'mismatch-badge'
        badge_text = '✓ 匹配' if is_match else '✗ 不匹配'
        
        details_html = ""
        if details:
            ssim_val = details.get('ssim', 0) * 100
            template_val = details.get('template_match', 0) * 100
            pixel_diff_val = details.get('pixel_diff', 0) * 100
            
            # 根據數值設置顏色
            def get_color(value, reverse=False):
                if reverse:
                    if value >= 90: return '#4CAF50'  # 綠色
                    elif value >= 70: return '#FFC107'  # 黃色
                    elif value >= 50: return '#FF9800'  # 橙色
                    else: return '#f44336'  # 紅色
                else:
                    if value <= 5: return '#4CAF50'  # 綠色（差異小）
                    elif value <= 15: return '#FFC107'  # 黃色
                    elif value <= 30: return '#FF9800'  # 橙色
                    else: return '#f44336'  # 紅色（差異大）
            
            details_html = f"""
                <div class="details">
                    <div class="detail-item">
                        <div class="detail-label">
                            <span class="detail-icon">📊</span> SSIM
                        </div>
                        <div class="detail-value-container">
                            <div class="detail-bar" style="width: {ssim_val}%; background: {get_color(ssim_val)};"></div>
                            <span class="detail-value">{ssim_val:.2f}%</span>
                        </div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">
                            <span class="detail-icon">🎯</span> 模板匹配
                        </div>
                        <div class="detail-value-container">
                            <div class="detail-bar" style="width: {template_val}%; background: {get_color(template_val)};"></div>
                            <span class="detail-value">{template_val:.2f}%</span>
                        </div>
                    </div>
                    <div class="detail-item">
                        <div class="detail-label">
                            <span class="detail-icon">🔍</span> 像素差異
                        </div>
                        <div class="detail-value-container">
                            <div class="detail-bar" style="width: {min(pixel_diff_val * 2, 100)}%; background: {get_color(pixel_diff_val, reverse=True)};"></div>
                            <span class="detail-value">{pixel_diff_val:.2f}%</span>
                        </div>
                    </div>
                </div>
            """
        
        html += f"""
                <tr>
                    <td>{i}</td>
                    <td>{timestamp_str}</td>
                    <td style="display:none;">
                        <div class="image-cell">
                            <a href="{image1_url}" target="_blank" title="點擊查看原圖">
                                <img src="{image1_url}" alt="{image1}" class="thumbnail" onerror="this.onerror=null; this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'150\' height=\'150\'%3E%3Crect fill=\'%23ddd\' width=\'150\' height=\'150\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' dy=\'.3em\' fill=\'%23999\' font-size=\'12\'%3E圖片載入失敗%3C/text%3E%3C/svg%3E';">
                            </a>
                            <span class="filename">{image1}</span>
                            <span class="path-text">{image1_url}</span>
                        </div>
                    </td>
                    <td style="display:none;">
                        <div class="image-cell">
                            <a href="{image2_url}" target="_blank" title="點擊查看原圖">
                                <img src="{image2_url}" alt="{image2}" class="thumbnail" onerror="this.onerror=null; this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'105\' height=\'105\'%3E%3Crect fill=\'%23ddd\' width=\'105\' height=\'105\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' dy=\'.3em\' fill=\'%23999\' font-size=\'11\'%3E圖片載入失敗%3C/text%3E%3C/svg%3E';">
                            </a>
                            <span class="filename">{image2}</span>
                            <span class="path-text">{image2_url}</span>
                        </div>
                    </td>
                    <td style="display:none;">
                        {_generate_image2_corrected_html(image2_corrected_path, image2_corrected_url, image2_corrected)}
                    </td>
                    <td><span class="{badge_class}">{badge_text}</span></td>
                    <td>
                        <div class="similarity-bar">
                            <div class="similarity-fill" style="width: {similarity}%"></div>
                        </div>
                        {similarity:.2f}%
                    </td>
                    <td>{threshold:.2f}%</td>
                    <td>{details_html}</td>
                    <td>{verification_html}</td>
                    <td>{overlay_html}</td>
                </tr>
        """
    
    # 將原本靜態表格的結尾換成可動態載入 comparison_log.json 的腳本
    html += f"""
            </tbody>
        </table>
        </div>
    </div>
    <script>
    // 內嵌當前記錄，若 fetch 失敗會使用此資料
    const inlineData = {json.dumps(records, ensure_ascii=False)};

    function normalizePath(p) {{
        if (!p) return '';
        let s = String(p);
        s = s.replace(/^\\/app\\//, '');
        s = s.replace(/^[A-Za-z]:[\\\\/]+/, '');
        s = s.replace(/\\\\\\\\/g, '/');
        // report.html 位於 logs/，若是相對路徑且未含 ../ ，補上 ../
        if (!s.startsWith('http://') && !s.startsWith('https://') && !s.startsWith('/') && !s.startsWith('../')) {{
            s = '../' + s;
        }}
        return s;
    }}

    function render(records) {{
        if (!Array.isArray(records) || records.length === 0) {{
            document.querySelector('.stat-card:nth-child(1) .stat-value').textContent = '0';
            document.querySelector('.stat-card:nth-child(2) .stat-value').textContent = '0';
            document.querySelector('.stat-card:nth-child(3) .stat-value').textContent = '-';
            document.querySelector('.stat-card:nth-child(4) .stat-value').textContent = '0';
            const tbody = document.querySelector('tbody');
            if (tbody) tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;color:#666;padding:20px;">尚無比對紀錄</td></tr>';
            return;
        }}

        // 按時間戳排序，最新的在最上面
        records.sort((a, b) => {{
            try {{
                const tsA = a.timestamp ? new Date(a.timestamp).getTime() : 0;
                const tsB = b.timestamp ? new Date(b.timestamp).getTime() : 0;
                return tsB - tsA; // 降序排列，最新的在前
            }} catch {{
                return 0;
            }}
        }});

        const matchCount = records.filter(r => r.is_match).length;
        const mismatchCount = records.length - matchCount;
        const avgSim = records.reduce((s, r) => s + (r.similarity || 0), 0) / records.length;

        document.querySelector('.stat-card:nth-child(1) .stat-value').textContent = matchCount;
        document.querySelector('.stat-card:nth-child(2) .stat-value').textContent = mismatchCount;
        document.querySelector('.stat-card:nth-child(3) .stat-value').textContent = (avgSim * 100).toFixed(2) + '%';
        document.querySelector('.stat-card:nth-child(4) .stat-value').textContent = records.length;

        const rows = records.map((r, idx) => {{
            const img1Url = normalizePath(r.image1);
            const img2Url = normalizePath(r.image2);
            const img2CorrectedUrl = r.image2_corrected ? normalizePath(r.image2_corrected) : null;
            const img1Name = img1Url.split('/').pop() || 'image1';
            const img2Name = img2Url.split('/').pop() || 'image2';
            const img2CorrectedName = img2CorrectedUrl ? img2CorrectedUrl.split('/').pop() || 'image2_corrected' : null;
            const sim = (r.similarity || 0) * 100;
            const threshold = (r.threshold || 0) * 100;
            const ssim = (r.details?.ssim || 0) * 100;
            const tmpl = (r.details?.template_match || 0) * 100;
            const diff = (r.details?.pixel_diff || 0) * 100;
            const badge = r.is_match ? '<span class="match-badge">✓ 匹配</span>' : '<span class="mismatch-badge">✗ 不匹配</span>';
            // 驗證圖像 URL（假設已生成）
            const comparisonUrl = `comparisons/comparison_${{idx + 1}}.jpg`;
            const heatmapUrl = `heatmaps/heatmap_${{idx + 1}}.jpg`;
            let ts = '-';
            if (r.timestamp) {{
                try {{
                    const date = new Date(r.timestamp);
                    const dateStr = date.toISOString().split('T')[0];
                    const timeStr = date.toTimeString().split(' ')[0].substring(0, 8);
                    ts = `${{dateStr}}<br>${{timeStr}}`;
                }} catch {{
                    // 如果無法解析，嘗試簡單分割
                    if (r.timestamp.includes(' ')) {{
                        const parts = r.timestamp.split(' ', 2);
                        ts = `${{parts[0]}}<br>${{parts[1]}}`;
                    }} else {{
                        ts = r.timestamp;
                    }}
                }}
            }}
            return `
                <tr>
                    <td>${{idx + 1}}</td>
                    <td>${{ts}}</td>
                    <td style="display:none;">
                        <div class="image-cell">
                            <a href="${{img1Url}}" target="_blank" title="點擊查看原圖">
                                <img src="${{img1Url}}" alt="${{img1Name}}" class="thumbnail" style="width:150px;height:150px;object-fit:contain;" onerror="this.style.display='none';">
                            </a>
                            <span class="filename">${{img1Name}}</span>
                            <span class="path-text">${{img1Url}}</span>
                        </div>
                    </td>
                    <td style="display:none;">
                        <div class="image-cell">
                            <a href="${{img2Url}}" target="_blank" title="點擊查看原圖">
                                <img src="${{img2Url}}" alt="${{img2Name}}" class="thumbnail" style="width:105px;height:105px;object-fit:contain;" onerror="this.onerror=null; this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'105\' height=\'105\'%3E%3Crect fill=\'%23ddd\' width=\'105\' height=\'105\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' dy=\'.3em\' fill=\'%23999\' font-size=\'11\'%3E圖片載入失敗%3C/text%3E%3C/svg%3E';">
                            </a>
                            <span class="filename">${{img2Name}}</span>
                            <span class="path-text">${{img2Url}}</span>
                        </div>
                    </td>
                    <td style="display:none;">
                        ${{img2CorrectedUrl ? `
                        <div class="image-cell">
                            <a href="${{img2CorrectedUrl}}" target="_blank" title="點擊查看原圖">
                                <img src="${{img2CorrectedUrl}}" alt="${{img2CorrectedName}}" class="thumbnail" style="width:105px;height:105px;object-fit:contain;" onerror="this.onerror=null; this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'105\' height=\'105\'%3E%3Crect fill=\'%23ddd\' width=\'105\' height=\'105\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' dy=\'.3em\' fill=\'%23999\' font-size=\'11\'%3E圖片載入失敗%3C/text%3E%3C/svg%3E';">
                            </a>
                            <span class="filename">${{img2CorrectedName}}</span>
                            <span class="path-text">${{img2CorrectedUrl}}</span>
                        </div>
                        ` : '<div style="color:#999;font-size:12px;">無校正圖像</div>'}}
                    </td>
                    <td>${{badge}}</td>
                    <td>
                        <div>${{sim.toFixed(2)}}%</div>
                        <div class="similarity-bar"><div class="similarity-fill" style="width: ${{sim}}%"></div></div>
                    </td>
                    <td>${{threshold.toFixed(2)}}%</td>
                    <td>
                        <div class="details">
                            <div class="detail-item">
                                <div class="detail-label">
                                    <span class="detail-icon">📊</span> SSIM
                                </div>
                                <div class="detail-value-container">
                                    <div class="detail-bar" style="width: ${{ssim}}%; background: ${{ssim >= 90 ? '#4CAF50' : ssim >= 70 ? '#FFC107' : ssim >= 50 ? '#FF9800' : '#f44336'}};"></div>
                                    <span class="detail-value">${{ssim.toFixed(2)}}%</span>
                                </div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">
                                    <span class="detail-icon">🎯</span> 模板匹配
                                </div>
                                <div class="detail-value-container">
                                    <div class="detail-bar" style="width: ${{tmpl}}%; background: ${{tmpl >= 90 ? '#4CAF50' : tmpl >= 70 ? '#FFC107' : tmpl >= 50 ? '#FF9800' : '#f44336'}};"></div>
                                    <span class="detail-value">${{tmpl.toFixed(2)}}%</span>
                                </div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-label">
                                    <span class="detail-icon">🔍</span> 像素差異
                                </div>
                                <div class="detail-value-container">
                                    <div class="detail-bar" style="width: ${{Math.min(diff * 2, 100)}}%; background: ${{diff <= 5 ? '#4CAF50' : diff <= 15 ? '#FFC107' : diff <= 30 ? '#FF9800' : '#f44336'}};"></div>
                                    <span class="detail-value">${{diff.toFixed(2)}}%</span>
                                </div>
                            </div>
                        </div>
                    </td>
                    <td>
                        <div class="verification-section">
                            <div class="verification-metrics">
                                <div class="metric-item">
                                    <span class="metric-label">旋轉角度:</span>
                                    <span class="metric-value">${{(r.details?.rotation_angle || 0).toFixed(2)}}°</span>
                                </div>
                                <div class="metric-item">
                                    <span class="metric-label">校正後:</span>
                                    <span class="metric-value">${{sim.toFixed(2)}}%</span>
                                </div>
                                ${{r.details?.similarity_before_correction ? `
                                <div class="metric-item">
                                    <span class="metric-label">校正前:</span>
                                    <span class="metric-value">${{(r.details.similarity_before_correction * 100).toFixed(2)}}%</span>
                                </div>
                                ` : ''}}
                                ${{r.details?.improvement !== undefined && r.details?.improvement !== null ? `
                                <div class="metric-item">
                                    <span class="metric-label">改善:</span>
                                    <span class="metric-value ${{r.details.improvement > 0 ? 'positive' : 'negative'}}">${{(r.details.improvement * 100).toFixed(2)}}%</span>
                                </div>
                                ` : ''}}
                            </div>
                            ${{comparisonUrl ? `
                            <div class="verification-image">
                                <a href="javascript:void(0)" onclick="openOverlayModal('comparisons/comparison_${{idx + 1}}.jpg', '校正前後對比圖')" title="點擊查看大圖">
                                    <img src="comparisons/comparison_${{idx + 1}}.jpg" alt="校正對比" class="verification-thumbnail" onerror="this.onerror=null;">
                                </a>
                                <div class="verification-label">並排對比</div>
                            </div>
                            ` : ''}}
                            ${{heatmapUrl ? `
                            <div class="verification-image">
                                <a href="javascript:void(0)" onclick="openOverlayModal('heatmaps/heatmap_${{idx + 1}}.jpg', '差異熱力圖')" title="點擊查看大圖">
                                    <img src="heatmaps/heatmap_${{idx + 1}}.jpg" alt="差異熱力圖" class="verification-thumbnail" onerror="this.onerror=null;">
                                </a>
                                <div class="verification-label">差異熱力圖</div>
                            </div>
                            ` : ''}}
                        </div>
                    </td>
                    <td>
                        <div class="overlay-section">
                            <div class="overlay-images">
                                <div class="overlay-item">
                                    <a href="javascript:void(0)" onclick="openOverlayModal('overlays/overlay_${{idx + 1}}_img1_on_img2.png', '圖像1(藍)疊在圖像2校正(紅)上<br>黃色=圖像2多出部分')" title="點擊查看大圖">
                                        <img src="overlays/overlay_${{idx + 1}}_img1_on_img2.png" alt="圖像1疊在圖像2校正上" class="overlay-thumbnail" style="width:98px;height:98px;object-fit:contain;" onerror="this.onerror=null; this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'98\' height=\'98\'%3E%3Crect fill=\'%23ddd\' width=\'98\' height=\'98\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' dy=\'.3em\' fill=\'%23999\' font-size=\'10\'%3E疊圖載入失敗%3C/text%3E%3C/svg%3E';">
                                    </a>
                                    <div class="overlay-label">圖像1(藍)疊在圖像2校正(紅)上<br>黃色=圖像2多出部分<br><span style="color:#666;font-size:10px;">點擊查看大圖</span></div>
                                </div>
                                <div class="overlay-item">
                                    <a href="javascript:void(0)" onclick="openOverlayModal('overlays/overlay_${{idx + 1}}_img2_on_img1.png', '圖像2校正(紅)疊在圖像1(藍)上<br>黃色=圖像1多出部分')" title="點擊查看大圖">
                                        <img src="overlays/overlay_${{idx + 1}}_img2_on_img1.png" alt="圖像2校正疊在圖像1上" class="overlay-thumbnail" style="width:98px;height:98px;object-fit:contain;" onerror="this.onerror=null; this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'98\' height=\'98\'%3E%3Crect fill=\'%23ddd\' width=\'98\' height=\'98\'/%3E%3Ctext x=\'50%25\' y=\'50%25\' text-anchor=\'middle\' dy=\'.3em\' fill=\'%23999\' font-size=\'10\'%3E疊圖載入失敗%3C/text%3E%3C/svg%3E';">
                                    </a>
                                    <div class="overlay-label">圖像2校正(紅)疊在圖像1(藍)上<br>黃色=圖像1多出部分<br><span style="color:#666;font-size:10px;">點擊查看大圖</span></div>
                                </div>
                            </div>
                        </div>
                    </td>
                </tr>
            `;
        }}).join('');

        const tbody = document.querySelector('tbody');
        if (tbody) tbody.innerHTML = rows;
    }}

    async function loadLatest() {{
        try {{
            const res = await fetch('comparison_log.json');
            if (res.ok) {{
                const data = await res.json();
                render(data);
                return;
            }}
        }} catch (e) {{}}
        // 若 fetch 失敗，改用內嵌資料
        render(inlineData);
    }}

    loadLatest();
    </script>
</body>
</html>
"""
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"✓ HTML 報告已生成: {output_file}")
    except IOError as e:
        print(f"錯誤：無法寫入報告檔案: {e}")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='生成印鑑比對測試報告',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='logs/comparison_log.json',
        help='記錄檔案路徑（預設: logs/comparison_log.json）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='輸出檔案路徑（不指定則輸出到終端）'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['text', 'html'],
        default='text',
        help='報告格式（預設: text）'
    )
    
    args = parser.parse_args()
    
    log_file = Path(args.log_file)
    records = load_comparison_logs(log_file)
    
    if not records:
        print("沒有比對記錄可生成報告")
        return
    
    if args.format == 'html':
        if not args.output:
            args.output = 'logs/report.html'
        generate_html_report(records, Path(args.output))
    else:
        output_file = Path(args.output) if args.output else None
        generate_text_report(records, output_file)


if __name__ == '__main__':
    main()
