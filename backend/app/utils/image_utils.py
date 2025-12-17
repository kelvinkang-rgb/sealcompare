"""
圖像處理工具函數
"""

import os
import uuid
from pathlib import Path
from typing import Optional
from fastapi import UploadFile
import shutil


def save_uploaded_file(upload_file: UploadFile, upload_dir: Path) -> tuple[Path, str]:
    """
    保存上傳的文件
    
    Args:
        upload_file: 上傳的文件對象
        upload_dir: 保存目錄
        
    Returns:
        (文件路徑, 文件名)
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成唯一文件名
    file_ext = Path(upload_file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = upload_dir / unique_filename
    
    # 保存文件
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return file_path, unique_filename


def get_file_size(file_path: Path) -> str:
    """
    獲取文件大小（人類可讀格式）
    
    Args:
        file_path: 文件路徑
        
    Returns:
        文件大小字符串
    """
    size = file_path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def delete_file(file_path: Path) -> bool:
    """
    刪除文件
    
    Args:
        file_path: 文件路徑
        
    Returns:
        是否成功刪除
    """
    try:
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    except Exception:
        return False

