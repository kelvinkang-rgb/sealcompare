"""
資料庫連接和會話管理
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

# 創建資料庫引擎
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# 創建會話工廠
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 創建基礎模型類
Base = declarative_base()


def get_db():
    """
    獲取資料庫會話（依賴注入）
    
    Yields:
        Session: 資料庫會話
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

