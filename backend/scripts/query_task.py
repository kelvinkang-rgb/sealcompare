#!/usr/bin/env python3
"""查詢 PDF 任務結果"""
import sys
from pathlib import Path

# 添加 backend 根目錄到 sys.path
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database import SessionLocal
from app.models import MultiSealComparisonTask
import json

task_uid = "91d5f5da-b5c2-42db-a96b-6dd5406ca8f2"

db = SessionLocal()
try:
    task = db.query(MultiSealComparisonTask).filter(
        MultiSealComparisonTask.task_uid == task_uid
    ).first()
    
    if not task:
        print(f"任務 {task_uid} 不存在")
        sys.exit(1)
    
    print(f"任務狀態: {task.status}")
    print(f"進度: {task.progress}%")
    print(f"\n結果結構:")
    
    if task.results:
        results = task.results
        if isinstance(results, dict):
            results_by_page = results.get("results_by_page", [])
            print(f"總頁數: {len(results_by_page)}")
            
            # 查找第一頁（page_index 可能是 0 或 1）
            page_0 = None
            for page in results_by_page:
                page_idx = page.get("page_index")
                print(f"  頁面 page_index={page_idx}, detected={page.get('detected')}, count={page.get('count')}")
                if page_idx == 0 or page_idx == 1:
                    if page_0 is None or page_idx < page_0.get("page_index", 999):
                        page_0 = page
            
            if page_0:
                print(f"\n第一頁 (page_index=0):")
                print(f"  page_image_id: {page_0.get('page_image_id')}")
                print(f"  detected: {page_0.get('detected')}")
                print(f"  count: {page_0.get('count')}")
                
                page_results = page_0.get("results", [])
                print(f"  印鑑數量: {len(page_results)}")
                
                # 查找「印鑑一」(seal_index=1)
                seal_1 = None
                for seal in page_results:
                    if seal.get("seal_index") == 1:
                        seal_1 = seal
                        break
                
                if seal_1:
                    print(f"\n  印鑑一 (seal_index=1) - 完整資料:")
                    print(json.dumps(seal_1, indent=4, ensure_ascii=False, default=str))
                    
                    print(f"\n  關鍵欄位:")
                    print(f"    similarity: {seal_1.get('similarity')}")
                    print(f"    alignment_angle: {seal_1.get('alignment_angle')}")
                    print(f"    alignment_offset: {seal_1.get('alignment_offset')}")
                    
                    # 提取 metrics（如果存在）
                    metrics = seal_1.get("metrics", {})
                    print(f"\n    Metrics keys: {list(metrics.keys()) if metrics else 'None'}")
                    if metrics:
                        print(f"\n    Metrics (完整):")
                        print(json.dumps(metrics, indent=6, ensure_ascii=False, default=str))
                        
                        alignment_metrics = metrics.get("alignment_optimization", {})
                        if alignment_metrics:
                            print(f"\n    Alignment Optimization:")
                            print(f"      final_offset_x: {alignment_metrics.get('final_offset_x')}")
                            print(f"      final_offset_y: {alignment_metrics.get('final_offset_y')}")
                            print(f"      final_angle: {alignment_metrics.get('final_angle')}")
                            print(f"      final_similarity: {alignment_metrics.get('final_similarity')}")
                            
                            # 檢查 pivot 相關
                            if "stage45_rotation_center" in alignment_metrics:
                                print(f"      stage45_rotation_center: {alignment_metrics.get('stage45_rotation_center')}")
                            if "stage45_initial_offset_from_pivot" in alignment_metrics:
                                print(f"      stage45_initial_offset_from_pivot: {alignment_metrics.get('stage45_initial_offset_from_pivot')}")
                            if "stage45_pivot1" in alignment_metrics:
                                print(f"      stage45_pivot1: {alignment_metrics.get('stage45_pivot1')}")
                            if "stage45_pivot2" in alignment_metrics:
                                print(f"      stage45_pivot2: {alignment_metrics.get('stage45_pivot2')}")
                            if "right_angle_base_rotation" in alignment_metrics:
                                print(f"      right_angle_base_rotation: {alignment_metrics.get('right_angle_base_rotation')}")
                            if "right_angle_fallback_used" in alignment_metrics:
                                print(f"      right_angle_fallback_used: {alignment_metrics.get('right_angle_fallback_used')}")
                else:
                    print(f"  未找到 seal_index=1 的印鑑")
            else:
                print(f"未找到 page_index=0 的頁面")
        else:
            print(f"結果格式不是 dict: {type(results)}")
    else:
        print("任務結果為空")
finally:
    db.close()

