"""
FastAPI 應用主入口
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from app.config import settings
from app.database import engine, Base
from app.api import images, comparisons, visualizations, statistics

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

