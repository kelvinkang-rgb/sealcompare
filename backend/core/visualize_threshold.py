"""
視覺化 threshold 對 pixel_diff_mask 的影響
展示不同 threshold 值如何影響像素差異的判定
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


def create_test_images(size=(400, 400)):
    """
    創建兩個測試圖像，用於演示 threshold 的效果
    
    Returns:
        gray1, gray2: 兩個灰度圖像
    """
    h, w = size
    
    # 圖像1：創建一個圓形印章（灰度值約150）
    gray1 = np.ones((h, w), dtype=np.uint8) * 255  # 白色背景
    center = (w // 2, h // 2)
    radius = min(w, h) // 3
    
    # 繪製圓形印章
    y, x = np.ogrid[:h, :w]
    mask_circle = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    gray1[mask_circle] = 150  # 印章顏色
    
    # 在印章內部添加一些文字或圖案（較暗的區域）
    inner_radius = radius * 0.6
    mask_inner = (x - center[0])**2 + (y - center[1])**2 <= inner_radius**2
    gray1[mask_inner] = 100  # 內部較暗
    
    # 圖像2：與圖像1相似，但有一些差異
    gray2 = gray1.copy()
    
    # 添加一些差異：
    # 1. 輕微的亮度變化（差異約20-30）
    variation_mask = mask_circle & (np.random.random((h, w)) > 0.7)
    gray2[variation_mask] = gray2[variation_mask] + np.random.randint(-30, 30, size=np.sum(variation_mask))
    gray2 = np.clip(gray2, 0, 255)
    
    # 2. 明顯的差異區域（差異約80-120）
    diff_region = (slice(center[1]-radius//2, center[1]+radius//2), 
                   slice(center[0]-radius//2, center[0]+radius//2))
    diff_region_h = diff_region[0].stop - diff_region[0].start
    diff_region_w = diff_region[1].stop - diff_region[1].start
    diff_mask = mask_circle[diff_region] & (np.random.random((diff_region_h, diff_region_w)) > 0.5)
    gray2[diff_region][diff_mask] = gray2[diff_region][diff_mask] + np.random.randint(80, 120, size=np.sum(diff_mask))
    gray2 = np.clip(gray2, 0, 255)
    
    # 3. 輕微差異區域（差異約10-20）
    light_diff_mask = mask_circle & (np.random.random((h, w)) > 0.85)
    gray2[light_diff_mask] = gray2[light_diff_mask] + np.random.randint(10, 20, size=np.sum(light_diff_mask))
    gray2 = np.clip(gray2, 0, 255)
    
    return gray1, gray2


def visualize_threshold_effect(gray1, gray2, threshold_values=[30, 50, 100, 150, 200], output_dir=None):
    """
    視覺化不同 threshold 值對 pixel_diff_mask 的影響
    
    Args:
        gray1: 第一個灰度圖像
        gray2: 第二個灰度圖像
        threshold_values: 要測試的 threshold 值列表
        output_dir: 輸出目錄
    """
    if output_dir is None:
        output_dir = Path("threshold_visualization")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 計算灰度差異
    gray_diff = cv2.absdiff(gray1, gray2)
    
    # 創建重疊區域mask（假設兩個圖像都有印章）
    h, w = gray1.shape
    center = (w // 2, h // 2)
    radius = min(w, h) // 3
    y, x = np.ogrid[:h, :w]
    overlap_mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    
    # 計算差異統計
    diff_stats = {
        'min': int(np.min(gray_diff[overlap_mask])) if np.any(overlap_mask) else 0,
        'max': int(np.max(gray_diff[overlap_mask])) if np.any(overlap_mask) else 0,
        'mean': float(np.mean(gray_diff[overlap_mask])) if np.any(overlap_mask) else 0.0,
        'median': float(np.median(gray_diff[overlap_mask])) if np.any(overlap_mask) else 0.0,
        'std': float(np.std(gray_diff[overlap_mask])) if np.any(overlap_mask) else 0.0,
    }
    
    print(f"\n灰度差異統計（在重疊區域內）：")
    print(f"  最小值: {diff_stats['min']}")
    print(f"  最大值: {diff_stats['max']}")
    print(f"  平均值: {diff_stats['mean']:.2f}")
    print(f"  中位數: {diff_stats['median']:.2f}")
    print(f"  標準差: {diff_stats['std']:.2f}")
    print(f"\n測試的 threshold 值: {threshold_values}")
    
    # 為每個 threshold 值創建視覺化
    fig_rows = len(threshold_values) + 1  # +1 用於原始圖像和差異圖
    fig = plt.figure(figsize=(16, 4 * fig_rows))
    
    # 第一行：原始圖像和差異圖
    ax1 = plt.subplot(fig_rows, 4, 1)
    ax1.imshow(gray1, cmap='gray', vmin=0, vmax=255)
    ax1.set_title('圖像1 (Gray1)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(fig_rows, 4, 2)
    ax2.imshow(gray2, cmap='gray', vmin=0, vmax=255)
    ax2.set_title('圖像2 (Gray2)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    ax3 = plt.subplot(fig_rows, 4, 3)
    im3 = ax3.imshow(gray_diff, cmap='hot', vmin=0, vmax=255)
    ax3.set_title(f'灰度差異圖 (Gray Diff)\n範圍: 0-255', fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, label='差異值')
    
    ax4 = plt.subplot(fig_rows, 4, 4)
    # 顯示差異值的直方圖
    diff_hist = gray_diff[overlap_mask].flatten()
    ax4.hist(diff_hist, bins=50, range=(0, 255), color='skyblue', edgecolor='black', alpha=0.7)
    ax4.axvline(x=100, color='red', linestyle='--', linewidth=2, label='預設 threshold=100')
    ax4.set_xlabel('差異值', fontsize=10)
    ax4.set_ylabel('像素數量', fontsize=10)
    ax4.set_title('差異值分佈直方圖', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 為每個 threshold 值創建視覺化
    for idx, threshold in enumerate(threshold_values):
        row = idx + 2  # 從第2行開始（第1行是原始圖像）
        
        # 計算 pixel_diff_mask
        pixel_diff_mask = (gray_diff > threshold) & overlap_mask
        
        # 計算統計資訊
        total_overlap_pixels = np.sum(overlap_mask)
        diff_pixels = np.sum(pixel_diff_mask)
        diff_ratio = diff_pixels / total_overlap_pixels if total_overlap_pixels > 0 else 0.0
        
        # 創建視覺化圖像
        vis_image = gray1.copy()
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # 標記差異區域（使用紅色）
        vis_image[pixel_diff_mask] = [0, 0, 255]  # BGR格式，紅色
        
        # 子圖1：顯示差異mask
        ax_mask = plt.subplot(fig_rows, 4, row * 4 - 3)
        mask_vis = np.zeros_like(gray_diff)
        mask_vis[pixel_diff_mask] = 255
        ax_mask.imshow(mask_vis, cmap='gray', vmin=0, vmax=255)
        ax_mask.set_title(f'Pixel Diff Mask\n(threshold={threshold})', fontsize=11, fontweight='bold')
        ax_mask.axis('off')
        
        # 添加統計文字
        stats_text = f'差異像素: {diff_pixels:,}\n比例: {diff_ratio*100:.2f}%'
        ax_mask.text(0.02, 0.98, stats_text, transform=ax_mask.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 子圖2：在原圖上標記差異區域
        ax_overlay = plt.subplot(fig_rows, 4, row * 4 - 2)
        ax_overlay.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        ax_overlay.set_title(f'原圖 + 差異標記\n(紅色=差異區域)', fontsize=11, fontweight='bold')
        ax_overlay.axis('off')
        
        # 子圖3：差異值的顏色映射（只顯示差異區域）
        ax_diff_colored = plt.subplot(fig_rows, 4, row * 4 - 1)
        diff_colored = gray_diff.copy()
        diff_colored[~pixel_diff_mask] = 0  # 非差異區域設為0
        im_diff = ax_diff_colored.imshow(diff_colored, cmap='hot', vmin=0, vmax=255)
        ax_diff_colored.set_title(f'差異值顏色映射\n(只顯示 > threshold)', fontsize=11, fontweight='bold')
        ax_diff_colored.axis('off')
        plt.colorbar(im_diff, ax=ax_diff_colored, label='差異值')
        
        # 子圖4：差異值分佈（標記threshold）
        ax_hist = plt.subplot(fig_rows, 4, row * 4)
        diff_hist = gray_diff[overlap_mask].flatten()
        ax_hist.hist(diff_hist, bins=50, range=(0, 255), color='skyblue', edgecolor='black', alpha=0.7)
        ax_hist.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'threshold={threshold}')
        
        # 標記差異區域（> threshold）
        diff_above_threshold = diff_hist[diff_hist > threshold]
        if len(diff_above_threshold) > 0:
            ax_hist.hist(diff_above_threshold, bins=50, range=(0, 255), color='red', alpha=0.5, label='差異區域')
        
        ax_hist.set_xlabel('差異值', fontsize=10)
        ax_hist.set_ylabel('像素數量', fontsize=10)
        ax_hist.set_title(f'差異值分佈\n(標記 threshold)', fontsize=11, fontweight='bold')
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.3)
        
        print(f"\nThreshold = {threshold}:")
        print(f"  差異像素數: {diff_pixels:,} / {total_overlap_pixels:,}")
        print(f"  差異比例: {diff_ratio*100:.2f}%")
    
    plt.tight_layout()
    output_file = output_dir / "threshold_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n視覺化結果已保存至: {output_file}")
    plt.close()
    
    # 創建單獨的對比圖（更清晰的版本）
    create_individual_comparisons(gray1, gray2, gray_diff, overlap_mask, threshold_values, output_dir)
    
    return output_dir


def create_individual_comparisons(gray1, gray2, gray_diff, overlap_mask, threshold_values, output_dir):
    """
    為每個 threshold 值創建單獨的對比圖
    """
    for threshold in threshold_values:
        pixel_diff_mask = (gray_diff > threshold) & overlap_mask
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # 左上：原始圖像並排
        ax1 = axes[0, 0]
        combined = np.hstack([gray1, gray2])
        ax1.imshow(combined, cmap='gray', vmin=0, vmax=255)
        ax1.set_title(f'原始圖像對比 (Threshold={threshold})', fontsize=14, fontweight='bold')
        ax1.axis('off')
        ax1.axvline(x=gray1.shape[1], color='yellow', linewidth=2, linestyle='--')
        ax1.text(gray1.shape[1]//2, -20, '圖像1', ha='center', fontsize=12, fontweight='bold')
        ax1.text(gray1.shape[1] + gray2.shape[1]//2, -20, '圖像2', ha='center', fontsize=12, fontweight='bold')
        
        # 右上：差異圖
        ax2 = axes[0, 1]
        im2 = ax2.imshow(gray_diff, cmap='hot', vmin=0, vmax=255)
        ax2.set_title('灰度差異圖 (0-255)', fontsize=14, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, label='差異值')
        
        # 添加 threshold 線
        contour = ax2.contour(gray_diff, levels=[threshold], colors='cyan', linewidths=2)
        ax2.clabel(contour, inline=True, fontsize=10, fmt=f'threshold={threshold}')
        
        # 左下：差異mask
        ax3 = axes[1, 0]
        mask_vis = np.zeros_like(gray_diff)
        mask_vis[pixel_diff_mask] = 255
        ax3.imshow(mask_vis, cmap='gray', vmin=0, vmax=255)
        ax3.set_title(f'Pixel Diff Mask (差異區域標記為白色)', fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # 添加統計資訊
        total_overlap = np.sum(overlap_mask)
        diff_pixels = np.sum(pixel_diff_mask)
        diff_ratio = diff_pixels / total_overlap if total_overlap > 0 else 0.0
        
        stats_text = (
            f'統計資訊:\n'
            f'重疊區域像素: {total_overlap:,}\n'
            f'差異像素: {diff_pixels:,}\n'
            f'差異比例: {diff_ratio*100:.2f}%\n'
            f'Threshold: {threshold}'
        )
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # 右下：原圖疊加差異標記
        ax4 = axes[1, 1]
        vis_image = gray1.copy()
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        vis_image[pixel_diff_mask] = [0, 0, 255]  # 紅色標記差異
        ax4.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        ax4.set_title('原圖 + 差異標記 (紅色區域)', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        output_file = output_dir / f"threshold_{threshold}_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  單獨對比圖已保存: {output_file}")
        plt.close()


def create_threshold_explanation_image(output_dir):
    """
    創建說明圖像，解釋 threshold 的概念
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 創建示例差異值分佈
    np.random.seed(42)
    diff_values = np.concatenate([
        np.random.normal(20, 5, 1000),   # 小差異
        np.random.normal(60, 10, 500),  # 中等差異
        np.random.normal(120, 15, 200), # 大差異
        np.random.normal(180, 20, 100)   # 很大差異
    ])
    diff_values = np.clip(diff_values, 0, 255)
    
    # 繪製直方圖
    n, bins, patches = ax.hist(diff_values, bins=50, range=(0, 255), 
                               color='skyblue', edgecolor='black', alpha=0.7)
    
    # 標記不同的 threshold 值
    thresholds = [30, 50, 100, 150, 200]
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    labels = ['很寬鬆 (30)', '寬鬆 (50)', '預設 (100)', '嚴格 (150)', '很嚴格 (200)']
    
    for threshold, color, label in zip(thresholds, colors, labels):
        ax.axvline(x=threshold, color=color, linestyle='--', linewidth=2.5, 
                  label=f'{label}: threshold={threshold}')
        
        # 標記該 threshold 以上的區域
        mask_above = diff_values > threshold
        if np.any(mask_above):
            above_values = diff_values[mask_above]
            ax.hist(above_values, bins=50, range=(0, 255), 
                   color=color, alpha=0.3, edgecolor=color, linewidth=1.5)
    
    ax.set_xlabel('灰度差異值 (Gray Diff)', fontsize=14, fontweight='bold')
    ax.set_ylabel('像素數量', fontsize=14, fontweight='bold')
    ax.set_title('Threshold 對 Pixel Diff Mask 的影響說明', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 添加說明文字
    explanation_text = (
        '說明:\n'
        '• gray_diff = |gray1 - gray2|，範圍: 0-255\n'
        '• threshold 範圍: 0-255\n'
        '• pixel_diff_mask = (gray_diff > threshold) & overlap_mask\n'
        '• threshold 越小，越多的像素被標記為差異\n'
        '• threshold 越大，只有明顯差異的像素被標記'
    )
    ax.text(0.02, 0.98, explanation_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_file = output_dir / "threshold_explanation.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"說明圖已保存: {output_file}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Threshold 視覺化工具")
    print("=" * 60)
    
    # 創建測試圖像
    print("\n1. 創建測試圖像...")
    gray1, gray2 = create_test_images(size=(400, 400))
    
    # 創建輸出目錄
    output_dir = Path("threshold_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存原始測試圖像
    cv2.imwrite(str(output_dir / "test_image1.png"), gray1)
    cv2.imwrite(str(output_dir / "test_image2.png"), gray2)
    print(f"   測試圖像已保存至: {output_dir}")
    
    # 創建說明圖
    print("\n2. 創建 threshold 說明圖...")
    create_threshold_explanation_image(output_dir)
    
    # 創建視覺化
    print("\n3. 創建 threshold 對比視覺化...")
    threshold_values = [30, 50, 100, 150, 200]
    visualize_threshold_effect(gray1, gray2, threshold_values, output_dir)
    
    print("\n" + "=" * 60)
    print("視覺化完成！")
    print(f"所有結果已保存至: {output_dir.absolute()}")
    print("=" * 60)

