# 印鑑比對（多印鑑 / PDF 全頁比對）

本專案提供 **多印鑑比對** 與 **PDF 全頁比對** 的完整流程（前端 UI + 後端 API + 非同步任務），以 `/multi-seal-test` 作為唯一主入口。

## 主要功能（以程式碼現況為準）

- **圖像1（模板）**：上傳後自動偵測印鑑位置，可手動微調並保存
- **圖像2（多印鑑 / PDF）**：可偵測多印鑑框並手動調整；PDF 可逐頁編輯
- **多印鑑比對（非同步任務）**：進度顯示；每顆印鑑回傳結果與視覺化產物（overlay/heatmap 等檔案）
- **PDF 全頁比對（非同步任務）**：將模板頁（圖像1）與圖像2 的所有頁面比對；提供全頁摘要、共用篩選器與 Histogram
- **去背景**：自動去除紙張背景並保留印泥筆劃（目前支援紅/藍印泥自動判別）

## 架構（Docker Compose）

- **Frontend**：Nginx（Port 3000）
- **Backend**：FastAPI（Port 8000）
- **PostgreSQL**：Port 5432
- **Redis**：Port 6379
- **Uploads/Logs**：Docker volume 持久化

## 快速開始（Makefile 唯一入口）

### 需求

- Docker Desktop（macOS/Windows）或 Docker（Linux）
- `make`

### 啟動

```bash
make up
```

### 使用

- **前端（唯一入口）**：`http://localhost:3000/multi-seal-test`
- **後端 API**：`http://localhost:8000`
- **API 文件**：`http://localhost:8000/docs`
- **健康檢查**：`http://localhost:8000/health`

### 日誌

```bash
make logs
```

### 停止

```bash
make down
```

## 測試

```bash
make test-backend
make test-e2e
```

## 主流程關鍵檔案（便於閱讀）

- **前端主頁面**：`frontend/src/pages/MultiSealTest.jsx`
- **前端結果呈現**：`frontend/src/components/MultiSealComparisonResults.jsx`
- **後端 API（主）**：`backend/app/api/images.py`
- **後端服務層**：`backend/app/services/image_service.py`
- **核心影像比對/去背景**：`backend/core/seal_compare.py`
- **E2E（Playwright）**：`frontend/e2e/pdf_image1_template_and_image2_multiseal.spec.js`


