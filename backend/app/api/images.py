"""
圖像相關 API
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List, Dict
from uuid import UUID
from pathlib import Path
import time
import logging
import traceback
import uuid as uuid_lib
from datetime import datetime

from app.database import get_db, SessionLocal
from app.schemas import (
    ImageResponse, 
    SealDetectionResponse, 
    SealLocationUpdate,
    MultipleSealsDetectionResponse,
    MultipleSealsSaveRequest,
    CropSealsRequest,
    CropSealsResponse,
    MultiSealComparisonRequest,
    MultiSealComparisonResponse,
    SealComparisonResult,
    RotatedSealsDetectionResponse,
    MultiSealComparisonTaskResponse,
    MultiSealComparisonTaskStatusResponse
)
from app.services.image_service import ImageService
from app.config import settings
from app.models import ComparisonStatus, MultiSealComparisonTask

# 配置日誌記錄器
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/images", tags=["images"])


@router.post("/upload", response_model=ImageResponse, status_code=201)
async def upload_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    上傳圖像文件
    
    - **file**: 圖像文件（支持 JPG, PNG, BMP 等格式）
    """
    # 檢查文件大小
    file_content = await file.read()
    if len(file_content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件大小超過限制 ({settings.MAX_UPLOAD_SIZE / 1024 / 1024}MB)"
        )
    
    # 重置文件指針
    file.file.seek(0)
    
    # 創建服務實例
    image_service = ImageService(db)
    
    try:
        image = image_service.create_image(file)
        return image
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上傳失敗: {str(e)}")


@router.get("/{image_id}", response_model=ImageResponse)
def get_image(
    image_id: UUID,
    db: Session = Depends(get_db)
):
    """
    獲取圖像信息
    
    - **image_id**: 圖像 ID
    """
    image_service = ImageService(db)
    image = image_service.get_image(image_id)
    
    if not image:
        raise HTTPException(status_code=404, detail="圖像不存在")
    
    return image


@router.get("/{image_id}/file")
def get_image_file(
    image_id: UUID,
    db: Session = Depends(get_db)
):
    """
    獲取圖像文件（返回實際圖像數據）
    
    - **image_id**: 圖像 ID
    """
    from fastapi.responses import FileResponse
    from pathlib import Path
    
    image_service = ImageService(db)
    image = image_service.get_image(image_id)
    
    if not image:
        raise HTTPException(status_code=404, detail="圖像不存在")
    
    file_path = Path(image.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="圖像文件不存在")
    
    return FileResponse(
        path=file_path,
        media_type=image.mime_type or "image/jpeg",
        filename=image.filename
    )


@router.delete("/{image_id}", status_code=204)
def delete_image(
    image_id: UUID,
    db: Session = Depends(get_db)
):
    """
    刪除圖像
    
    - **image_id**: 圖像 ID
    """
    image_service = ImageService(db)
    success = image_service.delete_image(image_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="圖像不存在")
    
    return None


@router.post("/{image_id}/detect-seal", response_model=SealDetectionResponse)
def detect_seal(
    image_id: UUID,
    db: Session = Depends(get_db)
):
    """
    檢測圖像中的印鑑位置
    
    - **image_id**: 圖像 ID
    """
    image_service = ImageService(db)
    try:
        result = image_service.detect_seal(image_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"檢測失敗: {str(e)}")


@router.put("/{image_id}/seal-location", response_model=ImageResponse)
def update_seal_location(
    image_id: UUID,
    location_data: SealLocationUpdate,
    db: Session = Depends(get_db)
):
    """
    更新用戶確認的印鑑位置
    
    - **image_id**: 圖像 ID
    - **bbox**: 邊界框 {"x": int, "y": int, "width": int, "height": int}
    - **center**: 中心點 {"center_x": int, "center_y": int, "radius": float}
    - **confidence**: 置信度（可選）
    """
    image_service = ImageService(db)
    try:
        image = image_service.update_seal_location(
            image_id,
            bbox=location_data.bbox,
            center=location_data.center,
            confidence=location_data.confidence
        )
        return image
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新失敗: {str(e)}")


@router.post("/{image_id}/detect-multiple-seals", response_model=MultipleSealsDetectionResponse)
def detect_multiple_seals(
    image_id: UUID,
    max_seals: int = Query(10, ge=1, le=160, description="最大檢測數量"),
    db: Session = Depends(get_db)
):
    """
    檢測圖像中的多個印鑑位置（測試功能）
    
    - **image_id**: 圖像 ID
    - **max_seals**: 最大檢測數量，默認10，範圍1-160
    """
    image_service = ImageService(db)
    try:
        result = image_service.detect_multiple_seals(image_id, max_seals=max_seals)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"檢測失敗: {str(e)}")


@router.post("/{image_id}/save-multiple-seals", response_model=ImageResponse)
def save_multiple_seals(
    image_id: UUID,
    request: MultipleSealsSaveRequest,
    db: Session = Depends(get_db)
):
    """
    保存多個印鑑位置到資料庫（測試功能）
    
    - **image_id**: 圖像 ID
    - **seals**: 印鑑列表，每個元素包含 bbox, center, confidence
    """
    image_service = ImageService(db)
    try:
        # 轉換為字典列表
        seals_data = [
            {
                'bbox': seal.bbox,
                'center': seal.center,
                'confidence': seal.confidence
            }
            for seal in request.seals
        ]
        image = image_service.save_multiple_seals(image_id, seals_data)
        return image
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存失敗: {str(e)}")


@router.post("/{image_id}/crop-seals", response_model=CropSealsResponse)
def crop_seals(
    image_id: UUID,
    request: CropSealsRequest,
    db: Session = Depends(get_db)
):
    """
    裁切圖像中的多個印鑑區域並保存為獨立圖像（測試功能）
    
    - **image_id**: 原圖像 ID
    - **seals**: 印鑑列表，每個元素包含 bbox, center, confidence
    - **margin**: 邊距（像素），默認15。邊距用於確保裁切後的圖像邊緣有足夠的背景區域，
                  供 remove_background_and_make_transparent 函數檢測背景色（至少需要5像素邊緣）
    """
    image_service = ImageService(db)
    try:
        # 轉換為字典列表
        seals_data = [
            {
                'bbox': seal.bbox,
                'center': seal.center,
                'confidence': seal.confidence
            }
            for seal in request.seals
        ]
        cropped_image_ids = image_service.crop_seals(
            image_id, 
            seals_data, 
            margin=request.margin or 15
        )
        return CropSealsResponse(
            cropped_image_ids=cropped_image_ids,
            count=len(cropped_image_ids)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"裁切失敗: {str(e)}")


@router.post("/{image1_id}/compare-with-seals", response_model=MultiSealComparisonTaskResponse)
def compare_image1_with_seals(
    image1_id: UUID,
    request: MultiSealComparisonRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    將圖像1與多個裁切的印鑑圖像進行比對（異步任務模式）
    
    - **image1_id**: 圖像1 ID
    - **seal_image_ids**: 裁切後的印鑑圖像 ID 列表
    - **threshold**: 相似度閾值，默認0.83
    
    立即返回任務 ID，比對在後台異步處理。
    使用 GET /images/tasks/{task_uid}/status 查詢任務狀態。
    """
    logger.info("=== 創建多印鑑比對任務 ===")
    logger.info(f"圖像1 ID: {image1_id}")
    logger.info(f"印鑑數量: {len(request.seal_image_ids)}")
    
    # 創建任務記錄（通過 Service 層）
    image_service = ImageService(db)
    task = image_service.create_multi_seal_comparison_task(
        image1_id=image1_id,
        seal_image_ids=request.seal_image_ids,
        threshold=request.threshold,
        similarity_ssim_weight=request.similarity_ssim_weight,
        similarity_template_weight=request.similarity_template_weight,
        pixel_similarity_weight=request.pixel_similarity_weight,
        histogram_similarity_weight=request.histogram_similarity_weight
    )
    
    logger.info(f"任務 UID: {task.task_uid}")
    
    # 提取mask權重參數（如果存在），否則使用預設值
    overlap_weight = request.overlap_weight if request.overlap_weight is not None else 0.5
    pixel_diff_penalty_weight = request.pixel_diff_penalty_weight if request.pixel_diff_penalty_weight is not None else 0.3
    unique_region_penalty_weight = request.unique_region_penalty_weight if request.unique_region_penalty_weight is not None else 0.2
    # 提取圖像對齊參數（如果存在），否則使用預設值
    rotation_range = request.rotation_range if request.rotation_range is not None else 15.0
    translation_range = request.translation_range if request.translation_range is not None else 100
    
    # 添加後台任務處理比對
    def process_comparison_task(task_uid_str: str, overlap_w: float, pixel_diff_penalty_w: float, unique_region_penalty_w: float, rotation_r: float, translation_r: int):
        """後台任務：處理比對"""
        db_task = SessionLocal()
        try:
            image_service_task = ImageService(db_task)
            try:
                task_record = image_service_task.get_multi_seal_comparison_task_or_raise(task_uid_str)
            except Exception as e:
                logger.error(f"任務不存在: {task_uid_str}, 錯誤: {str(e)}")
                return
            
            # 更新狀態為處理中
            task_record.status = ComparisonStatus.PROCESSING
            task_record.started_at = datetime.utcnow()
            task_record.progress = 0.0
            task_record.progress_message = "開始比對處理"
            db_task.commit()
            logger.info(f"任務開始處理: {task_uid_str}")
            
            # 轉換 seal_image_ids 回 UUID
            seal_image_ids = [UUID(sid) for sid in task_record.seal_image_ids]
            
            # 執行比對（這裡會花費較長時間）
            image_service = ImageService(db_task)
            
            # 更新進度：10%
            task_record.progress = 10.0
            task_record.progress_message = f"開始比對 {len(seal_image_ids)} 個印鑑"
            task_record.results = []  # 初始化結果列表
            task_record.total_count = len(seal_image_ids)
            db_task.commit()
            
            # 定義回調函數，當每個印鑑比對完成時立即更新任務記錄
            def update_task_with_result(result: Dict, current_index: int, total_count: int):
                """回調函數：當每個印鑑比對完成時更新任務記錄（線程安全）"""
                max_retries = 3
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        # 使用 with_for_update() 鎖定任務記錄，確保線程安全
                        # 這會阻止其他線程同時讀取和修改同一條記錄
                        task = db_task.query(MultiSealComparisonTask).filter(
                            MultiSealComparisonTask.task_uid == task_uid_str
                        ).with_for_update().first()
                        
                        if not task:
                            logger.warning(f"任務不存在，無法更新: {task_uid_str}")
                            return
                        
                        # 轉換結果為 JSON 格式
                        result_json = {
                            'seal_index': result['seal_index'],
                            'seal_image_id': str(result['seal_image_id']),
                            'similarity': result['similarity'],
                            'is_match': result['is_match'],
                            'overlay1_path': result['overlay1_path'],
                            'overlay2_path': result['overlay2_path'],
                            'heatmap_path': result['heatmap_path'],
                            'overlap_mask_path': result.get('overlap_mask_path'),
                            'pixel_diff_mask_path': result.get('pixel_diff_mask_path'),
                            'diff_mask_2_only_path': result.get('diff_mask_2_only_path'),
                            'diff_mask_1_only_path': result.get('diff_mask_1_only_path'),
                            'gray_diff_path': result.get('gray_diff_path'),
                            'mask_statistics': result.get('mask_statistics'),
                            'mask_based_similarity': result.get('mask_based_similarity'),
                            'structure_similarity': result.get('structure_similarity'),
                            'alignment_metrics': result.get('alignment_metrics'),
                            'input_image1_path': result.get('input_image1_path'),
                            'input_image2_path': result.get('input_image2_path'),
                            'error': result['error'],
                            'overlay_error': result.get('overlay_error'),
                            'timing': result.get('timing')  # 添加時間追蹤數據
                        }
                        
                        # 獲取當前結果列表（在鎖定狀態下讀取，確保是最新數據）
                        current_results = task.results or []
                        
                        # 使用字典追蹤已存在的結果（以 seal_index 為鍵），提高合併效率
                        results_dict = {r.get('seal_index'): r for r in current_results}
                        
                        # 更新或添加新結果
                        results_dict[result['seal_index']] = result_json
                        
                        # 轉換回列表並按 seal_index 排序
                        current_results = [results_dict[key] for key in sorted(results_dict.keys())]
                        
                        # 計算進度和成功數量
                        completed_count = len(current_results)
                        success_count = sum(1 for r in current_results if r.get('error') is None)
                        progress = 10.0 + (completed_count / total_count) * 80.0  # 10% 到 90%
                        
                        # 更新任務記錄
                        task.results = current_results
                        task.progress = progress
                        task.progress_message = f"已完成 {completed_count}/{total_count} 個印鑑比對（成功: {success_count}）"
                        task.success_count = success_count
                        db_task.commit()
                        
                        logger.info(f"任務更新: {task_uid_str}, 進度: {progress:.1f}%, 已完成: {completed_count}/{total_count}, 成功: {success_count}")
                        return  # 成功，退出重試循環
                        
                    except Exception as e:
                        retry_count += 1
                        db_task.rollback()
                        
                        if retry_count < max_retries:
                            # 等待一小段時間後重試（指數退避）
                            import time
                            wait_time = 0.1 * (2 ** (retry_count - 1))
                            logger.warning(f"更新任務記錄時出錯（重試 {retry_count}/{max_retries}）: {e}, 等待 {wait_time:.2f} 秒後重試")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"更新任務記錄時出錯（已重試 {max_retries} 次）: {e}")
                            import traceback
                            logger.error(f"錯誤堆疊:\n{traceback.format_exc()}")
            
            # 執行比對，傳入回調函數
            results_data = image_service.compare_image1_with_seals(
                task_record.image1_id,
                seal_image_ids,
                threshold=task_record.threshold,
                similarity_ssim_weight=task_record.similarity_ssim_weight,
                similarity_template_weight=task_record.similarity_template_weight,
                pixel_similarity_weight=task_record.pixel_similarity_weight,
                histogram_similarity_weight=task_record.histogram_similarity_weight,
                overlap_weight=overlap_w,
                pixel_diff_penalty_weight=pixel_diff_penalty_w,
                unique_region_penalty_weight=unique_region_penalty_w,
                rotation_range=rotation_r,
                translation_range=translation_r,
                task_uid=task_uid_str,
                task_update_callback=update_task_with_result
            )
            
            # 提取任務級別的時間數據
            task_timing = results_data.get('task_timing', {}) if isinstance(results_data, dict) else {}
            
            # 最終更新任務為完成前，驗證結果數量
            # 重新獲取任務記錄以確保獲取最新數據（回調函數可能已更新）
            task_record = image_service_task.get_multi_seal_comparison_task(task_uid_str)
            
            if task_record:
                current_results = task_record.results or []
                expected_count = len(seal_image_ids)
                actual_count = len(current_results)
                
                logger.info(f"[任務層] 開始驗證結果數量: 預期 {expected_count} 個，實際 {actual_count} 個")
                
                # 驗證結果數量是否正確
                if actual_count != expected_count:
                    logger.warning(f"[任務層] 結果數量不匹配！預期 {expected_count} 個，實際 {actual_count} 個")
                    
                    # 檢查缺失的印鑑索引
                    result_indices = {r.get('seal_index') for r in current_results if r.get('seal_index') is not None}
                    expected_indices = set(range(1, expected_count + 1))
                    missing_indices = sorted(expected_indices - result_indices)
                    
                    if missing_indices:
                        logger.warning(f"[任務層] 發現缺失的印鑑索引: {missing_indices}")
                        
                        # 識別缺失的印鑑 ID
                        missing_seal_image_ids = []
                        missing_seal_indices_list = []
                        for missing_idx in missing_indices:
                            if missing_idx <= len(seal_image_ids):
                                missing_seal_id = seal_image_ids[missing_idx - 1]
                                missing_seal_image_ids.append(missing_seal_id)
                                missing_seal_indices_list.append(missing_idx)
                                logger.info(f"[任務層] 缺失印鑑 {missing_idx}: ID={missing_seal_id}")
                        
                        # 重新比對缺失的印鑑
                        if missing_seal_image_ids:
                            logger.info(f"[任務層] 開始重新比對 {len(missing_seal_image_ids)} 個缺失的印鑑")
                            
                            # 更新進度訊息
                            task_record.progress = 90.0
                            task_record.progress_message = f"重新比對 {len(missing_seal_image_ids)} 個缺失的印鑑..."
                            db_task.commit()
                            
                            try:
                                # 調用重新比對功能
                                retry_results = image_service.retry_missing_seals(
                                    image1_id=task_record.image1_id,
                                    missing_seal_image_ids=missing_seal_image_ids,
                                    missing_seal_indices=missing_seal_indices_list,
                                    threshold=task_record.threshold,
                                    similarity_ssim_weight=task_record.similarity_ssim_weight,
                                    similarity_template_weight=task_record.similarity_template_weight,
                                    pixel_similarity_weight=task_record.pixel_similarity_weight,
                                    histogram_similarity_weight=task_record.histogram_similarity_weight,
                                    overlap_weight=overlap_w,
                                    pixel_diff_penalty_weight=pixel_diff_penalty_w,
                                    unique_region_penalty_weight=unique_region_penalty_w,
                                    rotation_range=rotation_r,
                                    translation_range=translation_r
                                )
                                
                                # 將重新比對的結果合併到現有結果中
                                # 使用字典追蹤結果（以 seal_index 為鍵）
                                results_dict = {r.get('seal_index'): r for r in current_results}
                                
                                # 添加重新比對的結果
                                for retry_result in retry_results:
                                    seal_idx = retry_result.get('seal_index')
                                    if seal_idx:
                                        # 轉換為 JSON 格式
                                        result_json = {
                                            'seal_index': retry_result['seal_index'],
                                            'seal_image_id': str(retry_result['seal_image_id']),
                                            'similarity': retry_result['similarity'],
                                            'is_match': retry_result['is_match'],
                                            'overlay1_path': retry_result['overlay1_path'],
                                            'overlay2_path': retry_result['overlay2_path'],
                                            'heatmap_path': retry_result['heatmap_path'],
                                            'overlap_mask_path': retry_result.get('overlap_mask_path'),
                                            'pixel_diff_mask_path': retry_result.get('pixel_diff_mask_path'),
                                            'diff_mask_2_only_path': retry_result.get('diff_mask_2_only_path'),
                                            'diff_mask_1_only_path': retry_result.get('diff_mask_1_only_path'),
                                            'gray_diff_path': retry_result.get('gray_diff_path'),
                                            'mask_statistics': retry_result.get('mask_statistics'),
                                            'mask_based_similarity': retry_result.get('mask_based_similarity'),
                                            'structure_similarity': retry_result.get('structure_similarity'),
                                            'alignment_metrics': retry_result.get('alignment_metrics'),
                                            'input_image1_path': retry_result.get('input_image1_path'),
                                            'input_image2_path': retry_result.get('input_image2_path'),
                                            'error': retry_result['error'],
                                            'overlay_error': retry_result.get('overlay_error'),
                                            'timing': retry_result.get('timing')  # 添加時間追蹤數據
                                        }
                                        results_dict[seal_idx] = result_json
                                
                                # 轉換回列表並按 seal_index 排序
                                current_results = [results_dict[key] for key in sorted(results_dict.keys())]
                                
                                logger.info(f"[任務層] 重新比對完成，合併後結果數量: {len(current_results)}")
                                
                                # 更新任務記錄
                                task_record.results = current_results
                                
                            except Exception as retry_error:
                                error_trace = traceback.format_exc()
                                logger.error(f"[任務層] 重新比對過程中發生錯誤: {retry_error}")
                                logger.error(f"[任務層] 錯誤堆疊:\n{error_trace}")
                                
                                # 為缺失的印鑑創建錯誤結果
                                results_dict = {r.get('seal_index'): r for r in current_results}
                                for missing_idx, missing_seal_id in zip(missing_seal_indices_list, missing_seal_image_ids):
                                    error_result = {
                                        'seal_index': missing_idx,
                                        'seal_image_id': str(missing_seal_id),
                                        'similarity': None,
                                        'is_match': None,
                                        'overlay1_path': None,
                                        'overlay2_path': None,
                                        'heatmap_path': None,
                                        'input_image1_path': None,
                                        'input_image2_path': None,
                                        'error': f"重新比對失敗: {str(retry_error)}",
                                        'overlay_error': None,
                                        'timing': {}  # 添加空的時間追蹤數據
                                    }
                                    results_dict[missing_idx] = error_result
                                
                                # 轉換回列表並排序
                                current_results = [results_dict[key] for key in sorted(results_dict.keys())]
                                task_record.results = current_results
                                
                                logger.warning(f"[任務層] 重新比對失敗，已為缺失的印鑑創建錯誤結果")
                
                # 最終驗證結果數量
                final_count = len(current_results)
                if final_count != expected_count:
                    logger.warning(f"[任務層] 最終結果數量仍不匹配: 預期 {expected_count} 個，實際 {final_count} 個")
                else:
                    logger.info(f"[任務層] 結果驗證通過: {final_count} 個結果")
                
                # 更新任務為完成
                success_count = sum(1 for r in current_results if r.get('error') is None)
                task_record.status = ComparisonStatus.COMPLETED
                task_record.completed_at = datetime.utcnow()
                task_record.progress = 100.0
                task_record.progress_message = f"比對完成：{success_count}/{expected_count} 個印鑑成功"
                task_record.success_count = success_count
                
                # 計算並保存任務級別的時間數據
                # 從 results 中計算 task_timing（如果還沒有從 results_data 中獲取）
                if task_timing is None or not task_timing:
                    valid_timings = [r.get('timing', {}) for r in current_results if r.get('timing') and r.get('error') is None]
                    if valid_timings:
                        total_times = [t.get('total', 0.0) for t in valid_timings if 'total' in t]
                        if total_times:
                            task_timing = {
                                'total_time': sum(total_times),
                                'parallel_processing_time': sum(total_times),  # 簡化處理
                                'average_seal_time': sum(total_times) / len(total_times) if total_times else 0.0
                            }
                            # 計算各個步驟的平均時間
                            step_keys = ['load_images', 'remove_bg_image1', 'remove_bg_align_image2', 'save_aligned_images',
                                       'similarity_calculation', 'save_corrected_images', 'create_overlay', 
                                       'calculate_mask_stats', 'create_heatmap']
                            for step_key in step_keys:
                                step_times = [t.get(step_key, 0.0) for t in valid_timings if step_key in t]
                                if step_times:
                                    task_timing[f'avg_{step_key}'] = sum(step_times) / len(step_times)
                            task_timing['avg_seal_total_time'] = sum(total_times) / len(total_times) if total_times else 0.0
                
                # 將 task_timing 存儲在 results 的第一個位置作為元數據（如果需要的話）
                # 但更好的方法是在 API 響應時計算，因為模型沒有專門的字段
                
                db_task.commit()
                
                logger.info(f"[任務層] 任務完成: {task_uid_str}, 成功: {success_count}/{expected_count}, 總結果數: {final_count}")
                if task_timing:
                    logger.info(f"[任務層] 任務時間統計: 總時間={task_timing.get('total_time', 0):.2f}秒, "
                              f"並行處理時間={task_timing.get('parallel_processing_time', 0):.2f}秒, "
                              f"平均每個印鑑時間={task_timing.get('average_seal_time', 0):.2f}秒")
            
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"任務處理失敗 {task_uid_str}: {str(e)}")
            logger.error(f"錯誤堆疊:\n{error_trace}")
            
            # 更新任務為失敗
            try:
                image_service_error = ImageService(db_task)
                task_record = image_service_error.get_multi_seal_comparison_task(task_uid_str)
                if task_record:
                    task_record.status = ComparisonStatus.FAILED
                    task_record.completed_at = datetime.utcnow()
                    task_record.error = str(e)
                    task_record.error_trace = error_trace
                    db_task.commit()
            except Exception as db_error:
                logger.error(f"更新失敗狀態時出錯: {str(db_error)}")
                db_task.rollback()
        finally:
            db_task.close()
    
    background_tasks.add_task(process_comparison_task, task.task_uid, overlap_weight, pixel_diff_penalty_weight, unique_region_penalty_weight, rotation_range, translation_range)
    
    logger.info(f"任務已創建並加入後台處理: {task.task_uid}")
    
    return MultiSealComparisonTaskResponse(
        id=task.id,
        task_uid=task.task_uid,
        image1_id=task.image1_id,
        status=task.status.value,
        progress=task.progress,
        progress_message=task.progress_message,
        results=None,
        total_count=None,
        success_count=None,
        error=None,
        error_trace=None,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at
    )


@router.get("/tasks/{task_uid}/status", response_model=MultiSealComparisonTaskStatusResponse)
def get_task_status(
    task_uid: str,
    db: Session = Depends(get_db)
):
    """
    查詢多印鑑比對任務狀態
    
    - **task_uid**: 任務 UID
    """
    image_service = ImageService(db)
    task = image_service.get_multi_seal_comparison_task_or_raise(task_uid)
    
    return MultiSealComparisonTaskStatusResponse(
        task_uid=task.task_uid,
        status=task.status.value,
        progress=task.progress,
        progress_message=task.progress_message,
        total_count=task.total_count,
        success_count=task.success_count,
        error=task.error,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at
    )


@router.get("/tasks/{task_uid}", response_model=MultiSealComparisonTaskResponse)
def get_task_result(
    task_uid: str,
    db: Session = Depends(get_db)
):
    """
    獲取多印鑑比對任務完整結果
    
    - **task_uid**: 任務 UID
    """
    image_service = ImageService(db)
    task = image_service.get_multi_seal_comparison_task_or_raise(task_uid)
    
    # 轉換結果為 SealComparisonResult 列表
    results = None
    task_timing = None
    if task.results:
        # 檢查 results 是否包含 task_timing（存儲在元數據中）
        # 如果 results 的第一個元素是字典且包含 'task_timing' 鍵，則提取它
        if isinstance(task.results, list) and len(task.results) > 0:
            # 嘗試從 results 中提取 task_timing（如果存在）
            # 實際 task_timing 應該從後台任務處理中獲取，這裡我們需要從數據庫或計算中獲取
            # 由於 task_timing 沒有持久化到數據庫，我們需要從 results 中計算或從其他地方獲取
            # 暫時設為 None，實際值應該在後台任務處理時計算並存儲
            pass
        
        results = [
            SealComparisonResult(**r)
            for r in task.results
        ]
    
    # 從後台任務處理中獲取 task_timing（如果有的話）
    # 由於 task_timing 沒有持久化，我們需要從 results 中計算平均值
    if results and task.status == ComparisonStatus.COMPLETED:
        valid_timings = [r.timing for r in results if r.timing and not r.error]
        if valid_timings:
            # 計算任務級別的時間統計
            total_times = [t.get('total', 0.0) for t in valid_timings if 'total' in t]
            if total_times:
                task_timing = {
                    'total_time': sum(total_times),
                    'parallel_processing_time': sum(total_times),  # 簡化處理，實際應該是並行時間
                    'average_seal_time': sum(total_times) / len(total_times) if total_times else 0.0
                }
                # 計算各個步驟的平均時間
                step_keys = ['load_images', 'remove_bg_image1', 'remove_bg_align_image2', 'save_aligned_images',
                           'similarity_calculation', 'save_corrected_images', 'create_overlay', 
                           'calculate_mask_stats', 'create_heatmap']
                for step_key in step_keys:
                    step_times = [t.get(step_key, 0.0) for t in valid_timings if step_key in t]
                    if step_times:
                        task_timing[f'avg_{step_key}'] = sum(step_times) / len(step_times)
                task_timing['avg_seal_total_time'] = sum(total_times) / len(total_times) if total_times else 0.0
    
    return MultiSealComparisonTaskResponse(
        id=task.id,
        task_uid=task.task_uid,
        image1_id=task.image1_id,
        status=task.status.value,
        progress=task.progress,
        progress_message=task.progress_message,
        results=results,
        total_count=task.total_count,
        success_count=task.success_count,
        error=task.error,
        error_trace=task.error_trace,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        task_timing=task_timing
    )


@router.get("/multi-seal-comparisons/{filename}")
def get_multi_seal_comparison_file(
    filename: str
):
    """
    獲取多印鑑比對視覺化文件（測試功能）
    
    - **filename**: 文件名（疊圖或熱力圖）
    """
    from app.config import settings
    
    file_path = Path(settings.LOGS_DIR) / "multi_seal_comparisons" / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    # 根據文件擴展名確定媒體類型
    if filename.endswith('.png'):
        media_type = "image/png"
    elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
        media_type = "image/jpeg"
    else:
        media_type = "image/png"
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )


@router.post("/{image1_id}/detect-matching-seals", response_model=RotatedSealsDetectionResponse)
def detect_matching_seals_with_rotation(
    image1_id: UUID,
    image2_id: UUID = Query(..., description="包含多個印鑑的圖像 ID"),
    rotation_range: float = Query(15.0, description="旋轉角度範圍（度）"),
    angle_step: float = Query(1.0, description="旋轉角度步長（度）"),
    max_seals: int = Query(10, description="最大返回數量"),
    db: Session = Depends(get_db)
):
    """
    檢測圖像2中與圖像1最相似的印鑑（考慮旋轉）
    
    - **image1_id**: 參考圖像 ID（模板）
    - **image2_id**: 包含多個印鑑的圖像 ID
    - **rotation_range**: 旋轉角度範圍（度），默認45度
    - **angle_step**: 旋轉角度步長（度），默認1度
    - **max_seals**: 最大返回數量，默認10個
    """
    image_service = ImageService(db)
    try:
        matches = image_service.detect_matching_seals_with_rotation(
            image1_id,
            image2_id,
            rotation_range=rotation_range,
            angle_step=angle_step,
            max_seals=max_seals
        )
        
        return RotatedSealsDetectionResponse(
            detected=len(matches) > 0,
            matches=matches,
            count=len(matches),
            reason=None if len(matches) > 0 else '未找到匹配的印鑑'
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"檢測失敗: {str(e)}")

