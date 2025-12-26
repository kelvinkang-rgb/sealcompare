"""
應用配置管理
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """應用設置"""
    
    # 應用基本設置
    APP_NAME: str = "印鑑比對系統"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # 資料庫設置
    DATABASE_URL: str = "postgresql://sealcompare:sealcompare@postgres:5432/sealcompare"
    
    # Redis 設置
    REDIS_URL: str = "redis://redis:6379/0"
    
    # 文件存儲設置
    UPLOAD_DIR: str = "/app/uploads"
    LOGS_DIR: str = "/app/logs"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 100MB（允許上傳接近原始文件大小的高解析度圖像）
    
    # CORS 設置
    CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:5173"]
    
    # API 設置
    API_V1_PREFIX: str = "/api/v1"
    
    # 多執行緒設置
    MAX_COMPARISON_THREADS: int = 48  # 多印鑑比對的最大線程數
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

