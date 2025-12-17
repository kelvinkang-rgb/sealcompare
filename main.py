"""
印鑑比對專案主程式
用於比對兩個印章圖像是否完全一致
"""

import argparse
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from seal_compare import SealComparator


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='印鑑比對工具 - 比對兩個印章圖像是否完全一致',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python main.py image1.jpg image2.jpg
  python main.py image1.jpg image2.jpg --threshold 0.98
        """
    )
    
    parser.add_argument(
        'image1',
        type=str,
        help='第一個印章圖像路徑'
    )
    
    parser.add_argument(
        'image2',
        type=str,
        help='第二個印章圖像路徑'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.95,
        help='相似度閾值 (0-1)，預設為 0.95 (95%%)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='顯示詳細比對資訊'
    )
    
    parser.add_argument(
        '--no-log',
        action='store_true',
        help='不保存比對記錄到檔案'
    )
    
    parser.add_argument(
        '--no-rotation',
        action='store_true',
        help='禁用旋轉角度搜索（加快速度但可能降低準確度）'
    )
    
    args = parser.parse_args()
    
    # 檢查文件是否存在
    if not Path(args.image1).exists():
        print(f"錯誤：找不到圖像文件 '{args.image1}'")
        sys.exit(1)
    
    if not Path(args.image2).exists():
        print(f"錯誤：找不到圖像文件 '{args.image2}'")
        sys.exit(1)
    
    # 檢查閾值範圍
    if not 0 <= args.threshold <= 1:
        print("錯誤：閾值必須在 0 到 1 之間")
        sys.exit(1)
    
    # 創建比對器
    comparator = SealComparator(threshold=args.threshold)
    
    # 執行比對
    print(f"正在比對圖像...")
    print(f"  圖像1: {args.image1}")
    print(f"  圖像2: {args.image2}")
    print(f"  相似度閾值: {args.threshold * 100:.1f}%")
    if not args.no_rotation:
        print(f"  旋轉角度搜索: 啟用（優化算法）")
    print()
    
    enable_rotation = not args.no_rotation
    is_match, similarity, details, img2_corrected = comparator.compare_files(
        args.image1, args.image2, enable_rotation_search=enable_rotation
    )
    
    # 顯示結果
    print("=" * 50)
    print("比對結果")
    print("=" * 50)
    
    if is_match:
        print("✓ 兩個印章圖像完全一致！")
    else:
        print("✗ 兩個印章圖像不一致")
    
    print(f"\n相似度: {similarity * 100:.2f}%")
    
    if args.verbose:
        print("\n詳細資訊:")
        print(f"  SSIM 相似度: {details.get('ssim', 0) * 100:.2f}%")
        print(f"  模板匹配度: {details.get('template_match', 0) * 100:.2f}%")
        print(f"  像素差異率: {details.get('pixel_diff', 0) * 100:.2f}%")
        print(f"  設定閾值: {details.get('threshold', 0) * 100:.2f}%")
        if details.get('rotation_angle') is not None:
            print(f"  最佳旋轉角度: {details.get('rotation_angle', 0):.2f}度")
    
    print("=" * 50)
    
    # 保存記錄到檔案
    if not args.no_log:
        save_comparison_log(args.image1, args.image2, is_match, similarity, details, args.threshold, img2_corrected)
    
    # 返回退出碼
    sys.exit(0 if is_match else 1)


def save_comparison_log(image1_path: str, image2_path: str, is_match: bool, 
                        similarity: float, details: dict, threshold: float, 
                        img2_corrected: Optional[np.ndarray] = None):
    """
    保存比對記錄到 JSON 檔案
    
    Args:
        image1_path: 第一個圖像路徑
        image2_path: 第二個圖像路徑
        is_match: 是否匹配
        similarity: 相似度
        details: 詳細資訊
        threshold: 閾值
        img2_corrected: 校正後的圖像2（可選）
    """
    import cv2
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "comparison_log.json"
    
    # 讀取現有記錄
    records = []
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
        except (json.JSONDecodeError, IOError):
            records = []
    
    # 保存校正後的圖像2（如果存在）
    image2_corrected_path = None
    if img2_corrected is not None and details.get('rotation_angle') is not None:
        corrected_dir = log_dir / "corrected_images"
        corrected_dir.mkdir(exist_ok=True)
        
        # 生成唯一檔名（基於時間戳）
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        corrected_filename = f"image2_corrected_{timestamp_str}.jpg"
        corrected_path = corrected_dir / corrected_filename
        
        cv2.imwrite(str(corrected_path), img2_corrected)
        image2_corrected_path = str(corrected_path.resolve())
    
    # 創建新記錄
    record = {
        'timestamp': datetime.now().isoformat(),
        'image1': str(Path(image1_path).resolve()),
        'image2': str(Path(image2_path).resolve()),
        'image2_corrected': image2_corrected_path,
        'is_match': is_match,
        'similarity': round(similarity, 4),
        'threshold': round(threshold, 4),
        'details': {
            'ssim': round(details.get('ssim', 0), 4),
            'template_match': round(details.get('template_match', 0), 4),
            'pixel_diff': round(details.get('pixel_diff', 0), 4),
            'rotation_angle': details.get('rotation_angle'),
            'similarity_before_correction': details.get('similarity_before_correction'),
            'improvement': details.get('improvement'),
            'image1_size': details.get('image1_size'),
            'image2_size': details.get('image2_size'),
            'size_ratio': details.get('size_ratio')
        }
    }
    
    # 添加新記錄
    records.append(record)
    
    # 保存到檔案
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 比對記錄已保存到: {log_file}")
    except IOError as e:
        print(f"\n警告：無法保存記錄到檔案: {e}")


if __name__ == '__main__':
    main()
