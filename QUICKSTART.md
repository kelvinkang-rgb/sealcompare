# 快速開始指南

## 方式一：使用 Docker Compose（最簡單）

### 1. 啟動所有服務

在專案根目錄執行：

**Windows (PowerShell):**
```powershell
docker-compose up -d --build
```

**Linux/Mac:**
```bash
docker-compose up -d --build
```

或者使用提供的啟動腳本：
- Windows: 雙擊 `start.bat`
- Linux/Mac: 執行 `bash start.sh`

### 2. 等待服務啟動

首次啟動需要一些時間來：
- 下載 Docker 映像
- 構建應用
- 初始化資料庫

查看服務狀態：
```bash
docker-compose ps
```

查看日誌：
```bash
docker-compose logs -f
```

### 3. 訪問應用

服務啟動完成後（約 1-2 分鐘），訪問：

- **前端界面**: http://localhost:3000
- **後端 API**: http://localhost:8000
- **API 文檔 (Swagger)**: http://localhost:8000/docs
- **API 文檔 (ReDoc)**: http://localhost:8000/redoc

### 4. 使用前端界面

1. 打開瀏覽器訪問 http://localhost:3000
2. 點擊「開始比對」或導航到「圖像比對」頁面
3. 上傳兩個印章圖像（圖像1 和圖像2）
4. 設置相似度閾值（預設 0.95）
5. 點擊「開始比對」
6. 等待處理完成（會顯示進度）
7. 查看比對結果和視覺化圖表

### 5. 查看比對記錄

- 點擊「比對記錄」查看歷史比對結果
- 或訪問 http://localhost:3000/history

## 方式二：本地開發模式

### 後端開發

1. **進入後端目錄**:
```bash
cd backend
```

2. **創建虛擬環境**:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **安裝依賴**:
```bash
pip install -r requirements.txt
```

4. **設置環境變數**:
```bash
# 複製示例文件
cp .env.example .env

# 編輯 .env 文件，確保資料庫連接正確
# DATABASE_URL=postgresql://sealcompare:sealcompare@localhost:5432/sealcompare
```

5. **確保 PostgreSQL 和 Redis 運行**:
```bash
# 使用 Docker 啟動資料庫服務
docker-compose up -d postgres redis
```

6. **初始化資料庫**:
```bash
# 資料庫表會在首次運行時自動創建
# 或手動運行（如果使用 Alembic）:
# alembic upgrade head
```

7. **啟動後端服務**:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

後端將在 http://localhost:8000 運行

### 前端開發

1. **進入前端目錄**:
```bash
cd frontend
```

2. **安裝依賴**:
```bash
npm install
```

3. **設置環境變數**（可選）:
創建 `frontend/.env` 文件：
```
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

4. **啟動開發服務器**:
```bash
npm run dev
```

前端將在 http://localhost:3000 運行（或 Vite 顯示的端口）

## 測試 API

### 使用 Swagger UI（推薦）

1. 訪問 http://localhost:8000/docs
2. 在 Swagger UI 中可以直接測試所有 API
3. 點擊「Try it out」按鈕
4. 填寫參數
5. 點擊「Execute」執行

### 使用 curl 命令

#### 1. 上傳圖像1
```bash
curl -X POST "http://localhost:8000/api/v1/images/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_images/seal_original_1.jpg"
```

保存返回的 `id` 作為 `image1_id`

#### 2. 上傳圖像2
```bash
curl -X POST "http://localhost:8000/api/v1/images/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_images/seal_rotated_medium.jpg"
```

保存返回的 `id` 作為 `image2_id`

#### 3. 創建比對任務
```bash
curl -X POST "http://localhost:8000/api/v1/comparisons/" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "image1_id": "YOUR_IMAGE1_ID",
    "image2_id": "YOUR_IMAGE2_ID",
    "threshold": 0.95,
    "enable_rotation_search": true
  }'
```

保存返回的 `id` 作為 `comparison_id`

#### 4. 查詢比對狀態
```bash
curl -X GET "http://localhost:8000/api/v1/comparisons/YOUR_COMPARISON_ID/status" \
  -H "accept: application/json"
```

#### 5. 獲取比對結果
```bash
curl -X GET "http://localhost:8000/api/v1/comparisons/YOUR_COMPARISON_ID" \
  -H "accept: application/json"
```

#### 6. 獲取視覺化圖像
```bash
# 並排對比圖
curl -O "http://localhost:8000/api/v1/comparisons/YOUR_COMPARISON_ID/comparison-image"

# 差異熱力圖
curl -O "http://localhost:8000/api/v1/comparisons/YOUR_COMPARISON_ID/heatmap"
```

## 常見問題

### 1. 端口已被占用

如果 3000 或 8000 端口已被占用，可以修改 `docker-compose.yml`:

```yaml
ports:
  - "3001:80"  # 前端改為 3001
  - "8001:8000"  # 後端改為 8001
```

### 2. 資料庫連接失敗

確保 PostgreSQL 服務正在運行：
```bash
docker-compose ps postgres
```

檢查資料庫連接字符串是否正確（在 `backend/.env` 中）

### 3. 前端無法連接到後端

檢查：
- 後端是否正在運行（http://localhost:8000）
- CORS 設置是否正確（在 `backend/app/config.py` 中）
- 前端環境變數 `VITE_API_BASE_URL` 是否正確

### 4. 圖像上傳失敗

檢查：
- 文件大小是否超過限制（預設 10MB）
- 文件格式是否支持（JPG, PNG, BMP 等）
- 上傳目錄權限是否正確

### 5. 比對處理失敗

查看後端日誌：
```bash
docker-compose logs backend
```

或本地運行時查看終端輸出

## 停止服務

```bash
docker-compose down
```

停止並刪除所有容器和網絡（保留數據卷）

```bash
docker-compose down -v
```

停止並刪除所有容器、網絡和數據卷

## 重新構建

如果修改了代碼，需要重新構建：

```bash
docker-compose up -d --build
```

## 查看日誌

查看所有服務日誌：
```bash
docker-compose logs -f
```

查看特定服務日誌：
```bash
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f postgres
```

## 下一步

- 查看完整的 API 文檔：http://localhost:8000/docs
- 閱讀 `README.md` 了解更多功能
- 查看代碼註釋了解實現細節

