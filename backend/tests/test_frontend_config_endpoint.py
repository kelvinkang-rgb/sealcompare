import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# pytest 在容器/本機的 rootdir 可能不同，明確把 backend 根目錄加到 sys.path
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import app  # noqa: E402


def test_frontend_config_endpoint_returns_defaults_when_yaml_missing(monkeypatch, tmp_path):
    # 指到不存在的 yaml，應回退到模型預設值
    missing = tmp_path / "missing.yml"
    monkeypatch.setenv("FRONTEND_CONFIG_PATH", str(missing))

    client = TestClient(app)
    r = client.get("/api/v1/config/frontend")
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["threshold_default"] == 0.5
    assert data["max_seals_default"] == 3
    assert data["rotation_range_default"] == 15.0
    assert data["translation_range_default"] == 100
    assert data["crop_margin_default"] == 10


def test_frontend_config_endpoint_reflects_yaml_override(monkeypatch, tmp_path):
    yml = tmp_path / "frontend.yml"
    yml.write_text(
        "\n".join(
            [
                "threshold_default: 0.62",
                "max_seals_default: 7",
                "rotation_range_default: 30.0",
                "translation_range_default: 200",
                "crop_margin_default: 12",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FRONTEND_CONFIG_PATH", str(yml))

    client = TestClient(app)
    r = client.get("/api/v1/config/frontend")
    assert r.status_code == 200, r.text
    data = r.json()

    assert data["threshold_default"] == 0.62
    assert data["max_seals_default"] == 7
    assert data["rotation_range_default"] == 30.0
    assert data["translation_range_default"] == 200
    assert data["crop_margin_default"] == 12


