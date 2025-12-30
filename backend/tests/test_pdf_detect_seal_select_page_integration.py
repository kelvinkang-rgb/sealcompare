import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# pytest 在容器/本機的 rootdir 可能不同，明確把 backend 根目錄加到 sys.path
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.main import app  # noqa: E402


def test_pdf_detect_seal_returns_best_page_and_persists_bbox():
    """
    整合測試（不 mock）：
    - 上傳真實 PDF
    - 呼叫 detect-seal（對 PDF 逐頁偵測）
    - 驗證回傳包含 page_image_id/page_index
    - 若偵測成功，對該 page_image_id 取回資料應已寫入 seal_bbox
    """
    repo_root = BACKEND_ROOT.parent
    pdf_path = repo_root / "test_images" / "案例一-印章有壓到線上.pdf"
    assert pdf_path.exists(), f"測試 PDF 不存在: {pdf_path}"

    client = TestClient(app)

    with pdf_path.open("rb") as f:
        r = client.post(
            "/api/v1/images/upload",
            files={"file": (pdf_path.name, f, "application/pdf")},
        )
    assert r.status_code == 201, r.text
    payload = r.json()
    assert payload.get("is_pdf") is True
    assert isinstance(payload.get("pages"), list)
    assert len(payload["pages"]) >= 1

    pdf_id = payload["id"]
    r2 = client.post(f"/api/v1/images/{pdf_id}/detect-seal")
    assert r2.status_code == 200, r2.text
    det = r2.json()

    # schema 應該永遠包含這兩個欄位（非 PDF 也會是 null）
    assert "page_image_id" in det
    assert "page_index" in det

    if det.get("detected") is True:
        assert det.get("bbox") is not None
        assert det.get("page_image_id") is not None
        assert det.get("page_index") is not None

        page_id = det["page_image_id"]
        r3 = client.get(f"/api/v1/images/{page_id}")
        assert r3.status_code == 200, r3.text
        page_payload = r3.json()
        assert page_payload.get("seal_bbox") is not None
    else:
        # 偵測失敗也應該有明確 reason
        assert det.get("bbox") is None
        assert det.get("reason") is not None


