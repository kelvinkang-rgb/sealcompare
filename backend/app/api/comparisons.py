"""
比對相關 API
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from app.database import get_db, SessionLocal
from app.schemas import ComparisonCreate, ComparisonResponse, ComparisonStatusResponse, ComparisonUpdate
from app.services.comparison_service import ComparisonService
from app.models import ComparisonStatus, Comparison

router = APIRouter(prefix="/comparisons", tags=["comparisons"])


@router.post("/", response_model=ComparisonResponse, status_code=201)
def create_comparison(
    comparison_data: ComparisonCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    創建比對任務（異步處理）
    
    - **image1_id**: 圖像1 ID
    - **image2_id**: 圖像2 ID
    - **threshold**: 相似度閾值（0-1，預設 0.95）
    - **enable_rotation_search**: 是否啟用旋轉角度搜索（預設 True）
    - **enable_translation_search**: 是否啟用平移搜索（預設 True，因為人工標記印鑑無法確保中心點都一致）
    """
    comparison_service = ComparisonService(db)
    
    try:
        # 創建比對記錄
        comparison = comparison_service.create_comparison(comparison_data)
        
        # 添加後台任務處理比對（在任務內部創建新的數據庫會話）
        def process_comparison_task(comp_id: UUID, enable_rot: bool, enable_trans: bool):
            """後台任務：處理比對（創建新的數據庫會話）"""
            db = SessionLocal()
            try:
                service = ComparisonService(db)
                service.process_comparison(comp_id, enable_rot, enable_trans)
            except Exception as e:
                # 記錄錯誤但不要讓任務失敗
                print(f"比對處理失敗 {comp_id}: {str(e)}")
                import traceback
                traceback.print_exc()
                # 更新狀態為失敗
                try:
                    db_comparison = db.query(Comparison).filter(Comparison.id == comp_id).first()
                    if db_comparison:
                        db_comparison.status = ComparisonStatus.FAILED
                        if db_comparison.details is None:
                            db_comparison.details = {}
                        db_comparison.details['error'] = str(e)
                        db_comparison.details['error_trace'] = traceback.format_exc()
                        db.commit()
                except Exception as db_error:
                    print(f"更新失敗狀態時出錯: {str(db_error)}")
                    db.rollback()
            finally:
                db.close()
        
        background_tasks.add_task(
            process_comparison_task,
            comparison.id,
            comparison_data.enable_rotation_search,
            comparison_data.enable_translation_search
        )
        
        return comparison
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"創建比對失敗: {str(e)}")


@router.get("/{comparison_id}", response_model=ComparisonResponse)
def get_comparison(
    comparison_id: UUID,
    db: Session = Depends(get_db)
):
    """
    獲取比對結果
    
    - **comparison_id**: 比對 ID
    """
    comparison_service = ComparisonService(db)
    comparison = comparison_service.get_comparison(comparison_id)
    
    if not comparison:
        raise HTTPException(status_code=404, detail="比對記錄不存在")
    
    return comparison


@router.get("/{comparison_id}/status", response_model=ComparisonStatusResponse)
def get_comparison_status(
    comparison_id: UUID,
    db: Session = Depends(get_db)
):
    """
    查詢比對狀態
    
    - **comparison_id**: 比對 ID
    """
    comparison_service = ComparisonService(db)
    comparison = comparison_service.get_comparison(comparison_id)
    
    if not comparison:
        raise HTTPException(status_code=404, detail="比對記錄不存在")
    
    # 從數據庫獲取進度信息（如果存在）
    progress = comparison.progress if hasattr(comparison, 'progress') and comparison.progress is not None else None
    message = comparison.progress_message if hasattr(comparison, 'progress_message') and comparison.progress_message else None
    current_step = comparison.current_step if hasattr(comparison, 'current_step') and comparison.current_step else None
    
    # 如果沒有進度信息，使用默認值
    if progress is None:
        if comparison.status == ComparisonStatus.PENDING:
            progress = 0
            message = message or "等待處理"
        elif comparison.status == ComparisonStatus.PROCESSING:
            progress = 50
            message = message or "正在處理中"
        elif comparison.status == ComparisonStatus.COMPLETED:
            progress = 100
            message = message or "處理完成"
        elif comparison.status == ComparisonStatus.FAILED:
            progress = 0
            message = message or "處理失敗"
    
    return ComparisonStatusResponse(
        id=comparison.id,
        status=comparison.status,
        progress=progress,
        message=message,
        current_step=current_step
    )


@router.get("/", response_model=List[ComparisonResponse])
def list_comparisons(
    skip: int = 0,
    limit: int = 100,
    include_deleted: bool = False,
    db: Session = Depends(get_db)
):
    """
    獲取比對記錄列表
    
    - **skip**: 跳過數量（用於分頁）
    - **limit**: 限制數量（用於分頁）
    - **include_deleted**: 是否包含已刪除的記錄（預設 False）
    """
    comparison_service = ComparisonService(db)
    comparisons = comparison_service.list_comparisons(skip=skip, limit=limit, include_deleted=include_deleted)
    return comparisons


@router.put("/{comparison_id}", response_model=ComparisonResponse)
def update_comparison(
    comparison_id: UUID,
    update_data: ComparisonUpdate,
    db: Session = Depends(get_db)
):
    """
    更新比對記錄
    
    - **comparison_id**: 比對 ID
    - **threshold**: 相似度閾值（可選）
    - **notes**: 備註或標籤（可選）
    - **status**: 狀態（可選，手動標記）
    """
    comparison_service = ComparisonService(db)
    
    try:
        # 轉換 Pydantic 模型為字典，只包含非 None 的值
        update_dict = update_data.model_dump(exclude_unset=True)
        comparison = comparison_service.update_comparison(comparison_id, update_dict)
        return comparison
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新比對記錄失敗: {str(e)}")


@router.delete("/{comparison_id}", status_code=204)
def delete_comparison(
    comparison_id: UUID,
    db: Session = Depends(get_db)
):
    """
    刪除比對記錄（軟刪除）
    
    - **comparison_id**: 比對 ID
    """
    comparison_service = ComparisonService(db)
    success = comparison_service.delete_comparison(comparison_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="比對記錄不存在")
    
    return None


@router.post("/{comparison_id}/restore", response_model=ComparisonResponse)
def restore_comparison(
    comparison_id: UUID,
    db: Session = Depends(get_db)
):
    """
    恢復已刪除的比對記錄
    
    - **comparison_id**: 比對 ID
    """
    comparison_service = ComparisonService(db)
    success = comparison_service.restore_comparison(comparison_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="比對記錄不存在或未被刪除")
    
    comparison = comparison_service.get_comparison(comparison_id, include_deleted=True)
    return comparison


@router.post("/{comparison_id}/retry", response_model=ComparisonResponse)
def retry_comparison(
    comparison_id: UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    enable_rotation_search: bool = Query(True, description="是否啟用旋轉角度搜索"),
    enable_translation_search: bool = Query(True, description="是否啟用平移搜索（預設開啟，因為人工標記印鑑無法確保中心點都一致）")
):
    """
    重新處理比對（用於失敗或卡住的比對）
    
    - **comparison_id**: 比對 ID
    - **enable_rotation_search**: 是否啟用旋轉角度搜索（預設 True）
    - **enable_translation_search**: 是否啟用平移搜索（預設 True，因為人工標記印鑑無法確保中心點都一致）
    """
    comparison_service = ComparisonService(db)
    comparison = comparison_service.get_comparison(comparison_id)
    
    if not comparison:
        raise HTTPException(status_code=404, detail="比對記錄不存在")
    
    # 重置狀態為 PENDING
    comparison.status = ComparisonStatus.PENDING
    comparison.details = {}  # 清除之前的錯誤信息
    db.commit()
    db.refresh(comparison)
    
    # 添加後台任務處理比對（在任務內部創建新的數據庫會話）
    def process_comparison_task(comp_id: UUID, enable_rot: bool, enable_trans: bool):
        """後台任務：處理比對（創建新的數據庫會話）"""
        db = SessionLocal()
        try:
            service = ComparisonService(db)
            service.process_comparison(comp_id, enable_rot, enable_trans)
        except Exception as e:
            # 記錄錯誤但不要讓任務失敗
            print(f"比對處理失敗 {comp_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            # 更新狀態為失敗
            try:
                db_comparison = db.query(Comparison).filter(Comparison.id == comp_id).first()
                if db_comparison:
                    db_comparison.status = ComparisonStatus.FAILED
                    if db_comparison.details is None:
                        db_comparison.details = {}
                    db_comparison.details['error'] = str(e)
                    db_comparison.details['error_trace'] = traceback.format_exc()
                    db.commit()
            except Exception as db_error:
                print(f"更新失敗狀態時出錯: {str(db_error)}")
                db.rollback()
        finally:
            db.close()
    
    background_tasks.add_task(
        process_comparison_task,
        comparison.id,
        enable_rotation_search,
        enable_translation_search
    )
    
    return comparison

