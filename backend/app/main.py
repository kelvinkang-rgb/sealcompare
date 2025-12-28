"""
FastAPI 應用主入口
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from app.config import settings
from app.database import engine, Base
from app.api import images, comparisons, visualizations, statistics
from app.exceptions import (
    ImageNotFoundError,
    ImageFileNotFoundError,
    InvalidBboxError,
    InvalidBboxSizeError,
    InvalidCenterError,
    InvalidSealDataError,
    ImageNotMarkedError,
    ImageReadError,
    CropAreaTooSmallError,
    ComparisonNotFoundError,
    VisualizationNotFoundError,
    MultiSealComparisonTaskNotFoundError
)

# 創建資料庫表
Base.metadata.create_all(bind=engine)

# 添加新欄位（如果不存在）- 用於向現有資料庫添加新欄位
def add_missing_columns():
    """添加缺失的欄位到現有表"""
    with engine.connect() as conn:
        # 檢查並添加 deleted_at 欄位
        try:
            conn.execute(text("""
                ALTER TABLE comparisons 
                ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMP
            """))
            conn.commit()
        except Exception as e:
            print(f"添加 deleted_at 欄位時出錯（可能已存在）: {e}")
            conn.rollback()
        
        # 檢查並添加 notes 欄位
        try:
            conn.execute(text("""
                ALTER TABLE comparisons 
                ADD COLUMN IF NOT EXISTS notes VARCHAR(500)
            """))
            conn.commit()
        except Exception as e:
            print(f"添加 notes 欄位時出錯（可能已存在）: {e}")
            conn.rollback()
        
        # 檢查並添加印鑑檢測相關欄位到 images 表
        try:
            conn.execute(text("""
                ALTER TABLE images 
                ADD COLUMN IF NOT EXISTS seal_detected BOOLEAN DEFAULT FALSE NOT NULL
            """))
            conn.commit()
        except Exception as e:
            print(f"添加 seal_detected 欄位時出錯（可能已存在）: {e}")
            conn.rollback()
        
        try:
            conn.execute(text("""
                ALTER TABLE images 
                ADD COLUMN IF NOT EXISTS seal_confidence REAL
            """))
            conn.commit()
        except Exception as e:
            print(f"添加 seal_confidence 欄位時出錯（可能已存在）: {e}")
            conn.rollback()
        
        try:
            conn.execute(text("""
                ALTER TABLE images 
                ADD COLUMN IF NOT EXISTS seal_bbox JSONB
            """))
            conn.commit()
        except Exception as e:
            print(f"添加 seal_bbox 欄位時出錯（可能已存在）: {e}")
            conn.rollback()
        
        try:
            conn.execute(text("""
                ALTER TABLE images 
                ADD COLUMN IF NOT EXISTS seal_center JSONB
            """))
            conn.commit()
        except Exception as e:
            print(f"添加 seal_center 欄位時出錯（可能已存在）: {e}")
            conn.rollback()
        
        # 檢查並添加 multiple_seals 欄位（測試功能）
        try:
            conn.execute(text("""
                ALTER TABLE images 
                ADD COLUMN IF NOT EXISTS multiple_seals JSONB
            """))
            conn.commit()
        except Exception as e:
            print(f"添加 multiple_seals 欄位時出錯（可能已存在）: {e}")
            conn.rollback()
        
        # 創建多印鑑比對任務表
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS multi_seal_comparison_tasks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    task_uid VARCHAR(36) UNIQUE NOT NULL,
                    image1_id UUID NOT NULL REFERENCES images(id),
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    progress REAL,
                    progress_message VARCHAR(500),
                    seal_image_ids JSONB NOT NULL,
                    threshold REAL NOT NULL DEFAULT 0.83,
                    similarity_ssim_weight REAL NOT NULL DEFAULT 0.5,
                    similarity_template_weight REAL NOT NULL DEFAULT 0.35,
                    pixel_similarity_weight REAL NOT NULL DEFAULT 0.1,
                    histogram_similarity_weight REAL NOT NULL DEFAULT 0.05,
                    results JSONB,
                    total_count INTEGER,
                    success_count INTEGER,
                    error VARCHAR(1000),
                    error_trace TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_task_uid ON multi_seal_comparison_tasks(task_uid)"))
            conn.commit()
        except Exception as e:
            print(f"創建 multi_seal_comparison_tasks 表時出錯（可能已存在）: {e}")
            conn.rollback()

# 執行遷移
try:
    add_missing_columns()
except Exception as e:
    print(f"資料庫遷移時出錯: {e}")

# 創建 FastAPI 應用
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="印鑑比對系統 API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 註冊路由
app.include_router(images.router, prefix=settings.API_V1_PREFIX)
app.include_router(comparisons.router, prefix=settings.API_V1_PREFIX)
app.include_router(visualizations.router, prefix=settings.API_V1_PREFIX)
app.include_router(statistics.router, prefix=settings.API_V1_PREFIX)


@app.get("/")
def root():
    """根路徑"""
    return {
        "message": "印鑑比對系統 API",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """健康檢查"""
    return {"status": "healthy"}


# 異常處理器
@app.exception_handler(ImageNotFoundError)
async def image_not_found_handler(request: Request, exc: ImageNotFoundError):
    """圖像不存在異常處理器"""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc) or "圖像不存在"}
    )


@app.exception_handler(ImageFileNotFoundError)
async def image_file_not_found_handler(request: Request, exc: ImageFileNotFoundError):
    """圖像文件不存在異常處理器"""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc) or "圖像文件不存在"}
    )


@app.exception_handler(InvalidBboxError)
async def invalid_bbox_handler(request: Request, exc: InvalidBboxError):
    """無效邊界框異常處理器"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc) or "邊界框格式錯誤"}
    )


@app.exception_handler(InvalidBboxSizeError)
async def invalid_bbox_size_handler(request: Request, exc: InvalidBboxSizeError):
    """邊界框尺寸太小異常處理器"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc) or "邊界框尺寸太小"}
    )


@app.exception_handler(InvalidCenterError)
async def invalid_center_handler(request: Request, exc: InvalidCenterError):
    """無效中心點異常處理器"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc) or "中心點格式錯誤"}
    )


@app.exception_handler(InvalidSealDataError)
async def invalid_seal_data_handler(request: Request, exc: InvalidSealDataError):
    """無效印鑑數據異常處理器"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc) or "印鑑數據格式錯誤"}
    )


@app.exception_handler(ImageNotMarkedError)
async def image_not_marked_handler(request: Request, exc: ImageNotMarkedError):
    """圖像未標記異常處理器"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc) or "圖像未標記印鑑位置，無法裁切"}
    )


@app.exception_handler(ImageReadError)
async def image_read_error_handler(request: Request, exc: ImageReadError):
    """圖像讀取錯誤異常處理器"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc) or "無法讀取圖像文件"}
    )


@app.exception_handler(CropAreaTooSmallError)
async def crop_area_too_small_handler(request: Request, exc: CropAreaTooSmallError):
    """裁切區域太小異常處理器"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc) or "裁切區域太小"}
    )


@app.exception_handler(ComparisonNotFoundError)
async def comparison_not_found_handler(request: Request, exc: ComparisonNotFoundError):
    """比對記錄不存在異常處理器"""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc) or "比對記錄不存在"}
    )


@app.exception_handler(VisualizationNotFoundError)
async def visualization_not_found_handler(request: Request, exc: VisualizationNotFoundError):
    """視覺化記錄不存在異常處理器"""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc) or "視覺化記錄不存在"}
    )


@app.exception_handler(MultiSealComparisonTaskNotFoundError)
async def task_not_found_handler(request: Request, exc: MultiSealComparisonTaskNotFoundError):
    """多印鑑比對任務不存在異常處理器"""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc) or "多印鑑比對任務不存在"}
    )

