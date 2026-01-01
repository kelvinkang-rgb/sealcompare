from fastapi import APIRouter

from app.frontend_config import load_frontend_config

router = APIRouter(tags=["config"])


@router.get("/config/frontend")
def get_frontend_config():
    """
    提供前端 runtime 可調預設值（由後端 config.yml 管理）。
    - 優先序固定：後端 YAML > 後端模型預設值
    """
    cfg = load_frontend_config()
    return cfg.model_dump()


