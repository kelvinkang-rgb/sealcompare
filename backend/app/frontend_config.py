from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field


class FrontendConfig(BaseModel):
    # 結構相似度判定閾值（structure_similarity >= threshold 視為匹配）
    threshold_default: float = Field(0.5, ge=0.0, le=1.0)
    # 多印鑑偵測/比對數量上限（UI 初始值）
    max_seals_default: int = Field(3, ge=1, le=160)

    # PDF/對齊預設（避免前端 hardcode）
    rotation_range_default: float = Field(15.0, ge=0.0, le=180.0)
    translation_range_default: int = Field(100, ge=0, le=1000)

    # 裁切邊距（像素）
    crop_margin_default: int = Field(10, ge=0, le=50)


def _default_frontend_yaml_path() -> Path:
    # backend/app/frontend_config.py -> backend/app -> backend
    backend_root = Path(__file__).resolve().parents[1]
    return backend_root / "config" / "frontend.yml"


def load_frontend_config() -> FrontendConfig:
    """
    載入後端提供給前端的 runtime config。

    優先序（固定 B）：
    - YAML（FRONTEND_CONFIG_PATH 或 backend/config/frontend.yml）
    - 各欄位的 model 預設值
    """
    cfg_path = os.getenv("FRONTEND_CONFIG_PATH")
    path = Path(cfg_path) if cfg_path else _default_frontend_yaml_path()

    data: Dict[str, Any] = {}
    try:
        if path.exists():
            import yaml  # PyYAML

            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(raw, dict):
                data = raw
    except Exception:
        # 任何解析錯誤都回退到預設值（避免 config 壞掉導致前端無法啟動）
        data = {}

    return FrontendConfig.model_validate(data)


