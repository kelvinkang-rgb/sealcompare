# 印鑑比對系統 - 前後端分離版本

這是一個現代化的印鑑比對系統，採用 FastAPI 後端 + React 前端的架構。

## 功能特點

- **RESTful API**: 完整的 API 接口，易於集成
- **前後端分離**: 前端和後端可獨立開發和部署
- **異步處理**: 圖像處理任務異步執行，不阻塞 API
- **現代化 UI**: 使用 React + Material-UI 構建
- **資料庫支持**: PostgreSQL 存儲比對記錄
- **自動文檔**: FastAPI 自動生成 API 文檔（Swagger）

## 技術棧

### 後端
- FastAPI - Web 框架
- SQLAlchemy - ORM
- PostgreSQL - 資料庫
- OpenCV - 圖像處理
- Celery + Redis - 異步任務（可選）

### 前端
- React 18 - UI 框架
- Material-UI - UI 組件庫
- React Query - 數據獲取和緩存
- Vite - 構建工具

## 快速開始

### 使用 Docker Compose（推薦）

1. **啟動所有服務**:
```bash
docker-compose up -d
```

2. **訪問應用**:
- 前端: http://localhost:3000
- 後端 API: http://localhost:8000
- API 文檔: http://localhost:8000/docs

3. **停止服務**:
```bash
docker-compose down
```

### 本地開發

#### 後端開發

1. **進入後端目錄**:
```bash
cd backend
```

2. **創建虛擬環境**:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **安裝依賴**:
```bash
pip install -r requirements.txt
```

4. **設置環境變數**:
```bash
cp .env.example .env
# 編輯 .env 文件
```

5. **初始化資料庫**:
```bash
# 確保 PostgreSQL 正在運行
alembic upgrade head
```

6. **啟動後端**:
```bash
uvicorn app.main:app --reload
```

#### 前端開發

1. **進入前端目錄**:
```bash
cd frontend
```

2. **安裝依賴**:
```bash
npm install
```

3. **啟動開發服務器**:
```bash
npm run dev
```

## API 文檔

啟動後端後，訪問 http://localhost:8000/docs 查看完整的 API 文檔。

### 主要 API 端點

- `POST /api/v1/images/upload` - 上傳圖像
- `GET /api/v1/images/{image_id}` - 獲取圖像信息
- `POST /api/v1/comparisons/` - 創建比對任務
- `GET /api/v1/comparisons/{comparison_id}` - 獲取比對結果
- `GET /api/v1/comparisons/{comparison_id}/status` - 查詢比對狀態
- `GET /api/v1/comparisons/{comparison_id}/comparison-image` - 獲取並排對比圖
- `GET /api/v1/comparisons/{comparison_id}/heatmap` - 獲取差異熱力圖
- `GET /api/v1/statistics/` - 獲取統計資訊

## 專案結構

```
sealcompare/
├── backend/                    # 後端服務
│   ├── app/
│   │   ├── api/               # API 路由
│   │   ├── services/          # 業務邏輯層
│   │   ├── models.py          # 資料庫模型
│   │   ├── schemas.py         # Pydantic 模型
│   │   └── main.py            # FastAPI 應用入口
│   ├── core/                  # 核心比對邏輯
│   ├── migrations/            # 資料庫遷移
│   └── requirements.txt       # 後端依賴
├── frontend/                  # 前端應用
│   ├── src/
│   │   ├── components/       # React 組件
│   │   ├── pages/             # 頁面組件
│   │   └── services/         # API 服務
│   └── package.json
└── docker-compose.yml         # Docker 編排配置
```

## 開發說明

### 資料庫遷移

使用 Alembic 進行資料庫遷移：

```bash
# 創建遷移
alembic revision --autogenerate -m "描述"

# 應用遷移
alembic upgrade head

# 回滾遷移
alembic downgrade -1
```

### 環境變數

後端環境變數配置在 `backend/.env` 文件中。主要配置項：

- `DATABASE_URL`: PostgreSQL 連接字符串
- `REDIS_URL`: Redis 連接字符串
- `UPLOAD_DIR`: 上傳文件存儲目錄
- `LOGS_DIR`: 日誌和生成文件目錄
- `CORS_ORIGINS`: 允許的 CORS 來源

## 授權

本專案僅供學習和研究使用。
