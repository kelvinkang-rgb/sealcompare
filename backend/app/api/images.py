"""
圖像相關 API
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from app.database import get_db
from app.schemas import ImageResponse
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

