# Frontend 環境變數設定說明

## 環境變數配置位置

### 1. 開發環境（本地開發時使用）

`.env` 文件應放在 `frontend/` 目錄下：

```
frontend/
├── .env                    # 開發環境配置（本地使用）
├── .env.example           # 配置範例文件
├── .env.local             # 本地覆蓋配置（可選，會覆蓋 .env）
└── .env.production        # 生產環境配置（可選）
```

#### 創建 .env 文件

複製範例文件：

```bash
cd frontend
cp .env.example .env
```

然後編輯 `.env` 文件，修改需要的配置。

**注意**：在開發環境中，如果使用 `npm run dev` 或 `vite` 啟動，`.env` 文件會自動被讀取。

---

### 2. Docker 構建環境（推薦方式）

在 Docker 構建時，環境變數是通過 `docker-compose.yml` 的 `build.args` 傳遞的。

#### 方式 A：直接在 docker-compose.yml 中設定（目前使用的方式）

編輯項目根目錄的 `docker-compose.yml`：

```yaml
frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile
    args:
      VITE_API_BASE_URL: /api/v1
      VITE_FEATURE_BATCH_SEAL_ADJUSTMENT: "true"
      VITE_FEATURE_METRICS_EXPLANATION_DIALOG: "false"  # 改為 false 隱藏
      VITE_FEATURE_SIMILARITY_HISTOGRAM: "false"        # 改為 false 隱藏
      # ... 其他配置
```

修改後，重新構建：

```bash
./rebuild-frontend.sh    # 或 rebuild-frontend.bat (Windows)
```

#### 方式 B：使用 .env 文件 + docker-compose.yml 引用

1. 在項目**根目錄**創建 `.env` 文件：

```bash
# 在項目根目錄（sealcompare/）
cat > .env << 'EOF'
VITE_FEATURE_METRICS_EXPLANATION_DIALOG=false
VITE_FEATURE_SIMILARITY_HISTOGRAM=false
VITE_API_BASE_URL=/api/v1
EOF
```

2. 修改 `docker-compose.yml`，使用環境變數引用：

```yaml
frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile
    args:
      VITE_API_BASE_URL: ${VITE_API_BASE_URL:-/api/v1}
      VITE_FEATURE_BATCH_SEAL_ADJUSTMENT: ${VITE_FEATURE_BATCH_SEAL_ADJUSTMENT:-"true"}
      VITE_FEATURE_METRICS_EXPLANATION_DIALOG: ${VITE_FEATURE_METRICS_EXPLANATION_DIALOG:-"true"}
      VITE_FEATURE_SIMILARITY_HISTOGRAM: ${VITE_FEATURE_SIMILARITY_HISTOGRAM:-"true"}
      # ... 其他配置
```

**注意**：docker-compose 會自動讀取項目根目錄的 `.env` 文件。

---

## 完整設定步驟範例

### 範例：隱藏 MetricsExplanationDialog 和 SimilarityHistogram

#### 步驟 1：編輯 docker-compose.yml

在項目根目錄編輯 `docker-compose.yml`：

```yaml
frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile
    args:
      VITE_API_BASE_URL: /api/v1
      # 功能開關配置
      VITE_FEATURE_BATCH_SEAL_ADJUSTMENT: "true"
      VITE_FEATURE_METRICS_EXPLANATION_DIALOG: "false"      # 隱藏指標說明對話框
      VITE_FEATURE_SIMILARITY_HISTOGRAM: "false"            # 隱藏相似度直方圖
      VITE_FEATURE_COMPARISON_EDIT_DIALOG: "true"
      VITE_FEATURE_DELETE_CONFIRM_DIALOG: "true"
      VITE_FEATURE_PROCESSING_STAGES: "true"
      VITE_FEATURE_VERIFICATION_VIEW: "true"
      VITE_FEATURE_IMAGE_MODAL: "true"
      VITE_FEATURE_IMAGE_PREVIEW_DIALOG: "true"
```

#### 步驟 2：重新構建前端服務

```bash
# Linux/macOS
./rebuild-frontend.sh

# Windows
rebuild-frontend.bat
```

#### 步驟 3：驗證

訪問 http://localhost:3000，確認對應的組件已被隱藏。

---

## 可用的環境變數

### API 配置

- `VITE_API_BASE_URL` - API 基礎路徑（預設：`/api/v1`）

### 功能開關（true = 顯示，false = 隱藏）

- `VITE_FEATURE_BATCH_SEAL_ADJUSTMENT` - 批量調整印鑑位置組件
- `VITE_FEATURE_METRICS_EXPLANATION_DIALOG` - 比對指標說明對話框
- `VITE_FEATURE_SIMILARITY_HISTOGRAM` - 相似度直方圖組件
- `VITE_FEATURE_COMPARISON_EDIT_DIALOG` - 比對記錄編輯對話框
- `VITE_FEATURE_DELETE_CONFIRM_DIALOG` - 刪除確認對話框
- `VITE_FEATURE_PROCESSING_STAGES` - 處理階段顯示組件
- `VITE_FEATURE_VERIFICATION_VIEW` - 校正驗證視圖組件
- `VITE_FEATURE_IMAGE_MODAL` - 圖像模態框組件
- `VITE_FEATURE_IMAGE_PREVIEW_DIALOG` - 圖像預覽對話框組件

---

## 注意事項

1. **Docker 構建時注入**：環境變數在構建時注入，修改後必須重新構建映像才能生效。

2. **開發環境 vs Docker 環境**：
   - 開發環境（`npm run dev`）：使用 `frontend/.env` 文件
   - Docker 環境：使用 `docker-compose.yml` 的 `build.args` 或項目根目錄的 `.env`

3. **預設值**：所有功能開關預設為 `"true"`（顯示），未設置時保持可見。

4. **字符串格式**：在 `docker-compose.yml` 中，值必須是字符串格式（用引號包圍）。

5. **重新構建**：修改配置後必須重新構建前端服務：
   ```bash
   ./rebuild-frontend.sh    # 或 rebuild-frontend.bat
   ```

---

## 快速參考

### 查看當前配置

查看 `docker-compose.yml` 中的 `frontend.build.args` 部分。

### 修改配置

1. 編輯 `docker-compose.yml` 中的 `frontend.build.args`
2. 執行 `./rebuild-frontend.sh`（或 `rebuild-frontend.bat`）
3. 訪問 http://localhost:3000 驗證

### 恢復預設配置

將 `docker-compose.yml` 中所有功能開關設為 `"true"`，然後重新構建。

