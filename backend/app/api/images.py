"""
圖像相關 API
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID
from pathlib import Path

from app.database import get_db
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
    RotatedSealsDetectionResponse
)
from app.services.image_service import ImageService
from app.config import settings

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
    db: Session = Depends(get_db)
):
    """
    檢測圖像中的多個印鑑位置（測試功能）
    
    - **image_id**: 圖像 ID
    """
    image_service = ImageService(db)
    try:
        result = image_service.detect_multiple_seals(image_id)
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
    - **margin**: 邊距（像素），默認10
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
            margin=request.margin or 10
        )
        return CropSealsResponse(
            cropped_image_ids=cropped_image_ids,
            count=len(cropped_image_ids)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"裁切失敗: {str(e)}")


@router.post("/{image1_id}/compare-with-seals", response_model=MultiSealComparisonResponse)
def compare_image1_with_seals(
    image1_id: UUID,
    request: MultiSealComparisonRequest,
    db: Session = Depends(get_db)
):
    """
    將圖像1與多個裁切的印鑑圖像進行比對（測試功能）
    
    - **image1_id**: 圖像1 ID
    - **seal_image_ids**: 裁切後的印鑑圖像 ID 列表
    - **threshold**: 相似度閾值，默認0.95
    """
    image_service = ImageService(db)
    try:
        results_data = image_service.compare_image1_with_seals(
            image1_id,
            request.seal_image_ids,
            threshold=request.threshold
        )
        
        # 轉換為響應格式
        results = [
            SealComparisonResult(
                seal_index=r['seal_index'],
                seal_image_id=r['seal_image_id'],
                similarity=r['similarity'],
                is_match=r['is_match'],
                overlay1_path=r['overlay1_path'],
                overlay2_path=r['overlay2_path'],
                heatmap_path=r['heatmap_path'],
                error=r['error']
            )
            for r in results_data
        ]
        
        success_count = sum(1 for r in results if r.error is None)
        
        return MultiSealComparisonResponse(
            image1_id=image1_id,
            results=results,
            total_count=len(results),
            success_count=success_count
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"比對失敗: {str(e)}")


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
    rotation_range: float = Query(45.0, description="旋轉角度範圍（度）"),
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

