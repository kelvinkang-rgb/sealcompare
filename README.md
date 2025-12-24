# 印鑑比對系統

一個現代化的印鑑比對系統，採用 FastAPI 後端 + React 前端的架構，提供高精度的印章圖像比對功能。

## 功能特點

### 核心功能

- **智能印鑑檢測**: 自動檢測圖像中的印鑑位置，支持手動調整和多印鑑檢測
- **高精度比對**: 採用多種相似度指標融合算法（SSIM、模板匹配、邊緣相似度等）
- **旋轉角度搜索**: 優化的三階段粗到細搜索策略，自動校正旋轉角度
- **平移校正**: 支持自動校正圖像平移，處理中心點不一致的情況
- **背景自動移除**: 自動檢測並移除圖像背景，提升比對準確度
- **圖像預處理**: CLAHE 亮度/對比度標準化，自適應二值化處理

### 系統功能

- **RESTful API**: 完整的 API 接口，易於集成
- **前後端分離**: 前端和後端可獨立開發和部署
- **異步處理**: 圖像處理任務異步執行，不阻塞 API 響應
- **現代化 UI**: 使用 React + Material-UI 構建，支持實時進度顯示
- **資料庫支持**: PostgreSQL 存儲比對記錄和圖像元數據
- **自動文檔**: FastAPI 自動生成 API 文檔（Swagger/ReDoc）
- **視覺化結果**: 提供並排對比圖、差異熱力圖、疊圖等多種視覺化
- **軟刪除機制**: 支持比對記錄的軟刪除和恢復
- **統計分析**: 提供比對統計資訊（匹配率、平均相似度等）
- **多印鑑測試**: 支持單張圖像中多個印鑑的檢測和比對

## 技術棧

### 後端

- **Python 3.11** - 運行環境
- **FastAPI** - 現代化 Web 框架
- **SQLAlchemy 2.0** - ORM 框架
- **PostgreSQL 15** - 關係型資料庫
- **Redis 7** - 緩存和任務隊列
- **OpenCV** - 圖像處理
- **NumPy** - 數值計算
- **Pillow** - 圖像處理
- **Uvicorn** - ASGI 服務器
- **Celery** - 異步任務處理（可選）

### 前端

- **React 18** - UI 框架
- **Material-UI (MUI)** - UI 組件庫
- **React Router** - 路由管理
- **React Query (TanStack Query)** - 數據獲取和緩存
- **Axios** - HTTP 客戶端
- **Recharts** - 圖表庫
- **Vite** - 構建工具
- **Nginx** - 生產環境 Web 服務器

## 系統架構

```
┌─────────────┐
│   Browser   │
└──────┬──────┘
       │
       │ HTTP
       ▼
┌─────────────┐      ┌─────────────┐
│   Frontend   │─────▶│   Backend   │
│  (Nginx)     │      │  (FastAPI)  │
│  Port: 3000  │      │  Port: 8000 │
└─────────────┘      └──────┬──────┘
                            │
                ┌───────────┼───────────┐
                │           │           │
                ▼           ▼           ▼
         ┌──────────┐ ┌──────────┐ ┌──────────┐
         │PostgreSQL│ │  Redis   │ │  Uploads │
         │  Port:   │ │  Port:   │ │  Volume  │
         │  5432    │ │  6379    │ │          │
         └──────────┘ └──────────┘ └──────────┘
```

## 快速開始

### 前置需求

- Docker Desktop（Windows/Mac）或 Docker + Docker Compose（Linux）
- 至少 4GB 可用記憶體
- 至少 2GB 可用磁碟空間

### 使用 Docker Compose（推薦）

這是最簡單的啟動方式，所有服務都會自動配置和啟動。

#### Windows

```powershell
# 方式一：使用啟動腳本
.\start.bat

# 方式二：直接使用 Docker Compose
docker-compose up -d --build
```

#### Linux/Mac

```bash
# 方式一：使用啟動腳本
bash start.sh

# 方式二：直接使用 Docker Compose
docker-compose up -d --build
```

#### 訪問應用

服務啟動完成後（約 1-2 分鐘），訪問：

- **前端界面**: http://localhost:3000
- **後端 API**: http://localhost:8000
- **API 文檔 (Swagger)**: http://localhost:8000/docs
- **API 文檔 (ReDoc)**: http://localhost:8000/redoc
- **健康檢查**: http://localhost:8000/health

#### 停止服務

```bash
docker-compose down
```

停止並刪除所有容器（保留數據卷）：

```bash
docker-compose down -v
```

停止並刪除所有容器和數據卷（**注意：會刪除所有數據**）

### 本地開發

#### 後端開發

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
創建 `backend/.env` 文件：
```env
DATABASE_URL=postgresql://sealcompare:sealcompare@localhost:5432/sealcompare
REDIS_URL=redis://localhost:6379/0
UPLOAD_DIR=./uploads
LOGS_DIR=./logs
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]
```

5. **啟動資料庫服務**（使用 Docker）:
```bash
docker-compose up -d postgres redis
```

6. **初始化資料庫**:
資料庫表會在首次運行時自動創建，或手動運行：
```bash
# 如果使用 Alembic
alembic upgrade head
```

7. **啟動後端**:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

後端將在 http://localhost:8000 運行

#### 前端開發

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
```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

4. **啟動開發服務器**:
```bash
npm run dev
```

前端將在 http://localhost:5173 運行（或 Vite 顯示的端口）

## 專案結構

```
sealcompare/
├── backend/                           # 後端服務
│   ├── app/
│   │   ├── api/                       # API 路由
│   │   │   ├── images.py             # 圖像管理 API
│   │   │   ├── comparisons.py        # 比對管理 API
│   │   │   ├── visualizations.py    # 視覺化 API
│   │   │   └── statistics.py         # 統計 API
│   │   ├── services/                 # 業務邏輯層
│   │   │   ├── image_service.py      # 圖像服務
│   │   │   └── comparison_service.py # 比對服務
│   │   ├── utils/                    # 工具類
│   │   │   ├── image_utils.py       # 圖像處理工具
│   │   │   └── seal_detector.py     # 印鑑檢測工具
│   │   ├── models.py                 # SQLAlchemy 資料庫模型
│   │   ├── schemas.py                # Pydantic 驗證模型
│   │   ├── config.py                 # 應用配置
│   │   ├── database.py               # 資料庫連接
│   │   └── main.py                   # FastAPI 應用入口
│   ├── core/                         # 核心比對邏輯
│   │   ├── seal_compare.py          # 主要比對算法
│   │   ├── overlay.py               # 圖像疊加處理
│   │   └── verification.py          # 驗證邏輯
│   ├── docs/                         # 文檔
│   │   ├── comparison_flow_diagram.svg  # 流程圖
│   │   └── FLOW_DIAGRAM_README.md   # 流程說明
│   ├── Dockerfile                    # 後端 Docker 配置
│   ├── requirements.txt              # Python 依賴
│   └── alembic.ini                   # Alembic 遷移配置
├── frontend/                         # 前端應用
│   ├── src/
│   │   ├── components/              # React 組件
│   │   │   ├── ComparisonForm.jsx   # 比對表單
│   │   │   ├── ComparisonResult.jsx # 比對結果展示
│   │   │   ├── ImagePreview.jsx     # 圖像預覽
│   │   │   ├── SealDetectionBox.jsx  # 印鑑檢測框
│   │   │   ├── MultiSealDetectionBox.jsx  # 多印鑑檢測框
│   │   │   └── ...                  # 其他組件
│   │   ├── pages/                   # 頁面組件
│   │   │   ├── Home.jsx            # 首頁
│   │   │   ├── Comparison.jsx      # 比對頁面
│   │   │   ├── History.jsx         # 歷史記錄頁面
│   │   │   └── MultiSealTest.jsx   # 多印鑑測試頁面
│   │   ├── services/
│   │   │   └── api.js              # API 服務封裝
│   │   ├── App.jsx                  # 應用入口
│   │   └── main.jsx                 # React 入口
│   ├── Dockerfile                    # 前端 Docker 配置（多階段構建）
│   ├── nginx.conf                    # Nginx 配置
│   ├── package.json                  # Node.js 依賴
│   └── vite.config.js                # Vite 配置
├── test_images/                      # 測試圖像文件
├── docker-compose.yml                # Docker Compose 編排配置
├── start.bat                         # Windows 啟動腳本
├── start.sh                          # Linux/Mac 啟動腳本
├── deploy.bat                        # Windows 部署腳本
├── deploy.sh                         # Linux/Mac 部署腳本
├── README.md                         # 專案說明文檔
├── QUICKSTART.md                     # 快速開始指南
└── VERIFY.md                         # 驗證步驟指南
```

## API 文檔

啟動後端後，訪問 http://localhost:8000/docs 查看完整的 API 文檔。

### 主要 API 端點

#### 圖像管理 API

- `POST /api/v1/images/upload` - 上傳圖像文件
- `GET /api/v1/images/` - 獲取圖像列表（支持分頁）
- `GET /api/v1/images/{image_id}` - 獲取圖像信息
- `GET /api/v1/images/{image_id}/file` - 獲取圖像文件（返回實際圖像數據）
- `DELETE /api/v1/images/{image_id}` - 刪除圖像
- `POST /api/v1/images/{image_id}/detect-seal` - 檢測圖像中的印鑑位置
- `PUT /api/v1/images/{image_id}/seal-location` - 更新用戶確認的印鑑位置
- `POST /api/v1/images/{image_id}/detect-multiple-seals` - 檢測圖像中的多個印鑑

#### 比對管理 API

- `POST /api/v1/comparisons/` - 創建比對任務（異步處理）
- `GET /api/v1/comparisons/` - 獲取比對記錄列表（支持分頁和過濾）
- `GET /api/v1/comparisons/{comparison_id}` - 獲取比對結果詳情
- `GET /api/v1/comparisons/{comparison_id}/status` - 查詢比對狀態和進度
- `PUT /api/v1/comparisons/{comparison_id}` - 更新比對記錄（備註、閾值等）
- `DELETE /api/v1/comparisons/{comparison_id}` - 刪除比對記錄（軟刪除）
- `POST /api/v1/comparisons/{comparison_id}/restore` - 恢復已刪除的比對記錄
- `POST /api/v1/comparisons/{comparison_id}/retry` - 重新處理比對任務

#### 視覺化 API

- `GET /api/v1/comparisons/{comparison_id}/comparison-image` - 獲取並排對比圖
- `GET /api/v1/comparisons/{comparison_id}/heatmap` - 獲取差異熱力圖
- `GET /api/v1/comparisons/{comparison_id}/overlay?overlay_type=1|2` - 獲取疊圖

#### 統計 API

- `GET /api/v1/statistics/` - 獲取統計資訊（總比對次數、匹配率、平均相似度等）

## 前端頁面

### 首頁 (Home)
- 系統介紹和快速導航
- 快速開始比對的入口

### 比對頁面 (Comparison)
- 上傳兩個圖像進行比對
- 自動檢測印鑑位置
- 手動調整印鑑區域
- 實時顯示比對進度
- 查看比對結果和視覺化圖表

### 歷史記錄 (History)
- 查看所有比對記錄
- 支持篩選和搜索
- 查看詳細比對結果
- 軟刪除和恢復功能

### 多印鑑測試 (MultiSealTest)
- 測試單張圖像中的多個印鑑檢測
- 批量比對功能

## Docker 配置說明

### 後端 Dockerfile

- **基礎映像**: `python:3.11-slim`
- **系統依賴**: 
  - OpenCV 所需庫（libgl1, libglib2.0-0）
  - 中文字體支持（fonts-wqy-zenhei, fonts-wqy-microhei）
- **端口**: 8000
- **啟動命令**: `uvicorn app.main:app --host 0.0.0.0 --port 8000`

### 前端 Dockerfile

- **構建階段**: `node:18-alpine`
  - 安裝依賴並構建 React 應用
- **生產階段**: `nginx:alpine`
  - 使用 Nginx 提供靜態文件服務
  - 配置 API 代理轉發
- **端口**: 80（映射到主機 3000）

### Docker Compose 服務

- **postgres**: PostgreSQL 15 資料庫
- **redis**: Redis 7 緩存服務
- **backend**: FastAPI 後端服務
- **frontend**: Nginx + React 前端服務

## 環境變數

### 後端環境變數

在 `docker-compose.yml` 或 `backend/.env` 中配置：

- `DATABASE_URL`: PostgreSQL 連接字符串（預設：`postgresql://sealcompare:sealcompare@postgres:5432/sealcompare`）
- `REDIS_URL`: Redis 連接字符串（預設：`redis://redis:6379/0`）
- `UPLOAD_DIR`: 上傳文件存儲目錄（預設：`/app/uploads`）
- `LOGS_DIR`: 日誌和生成文件目錄（預設：`/app/logs`）
- `CORS_ORIGINS`: 允許的 CORS 來源（JSON 格式的列表）

### 前端環境變數

在構建時通過 Dockerfile ARG 傳遞：

- `VITE_API_BASE_URL`: API 基礎 URL（預設：`/api/v1`）

## 比對算法說明

系統採用多種相似度指標融合算法，主要包括：

1. **SSIM (結構相似性指數)**: 衡量圖像結構相似度
2. **模板匹配**: 使用 OpenCV 模板匹配算法
3. **像素相似度**: 計算像素級差異
4. **直方圖相似度**: 比較圖像直方圖分佈
5. **邊緣相似度**: 自適應邊緣檢測和比較
6. **精確匹配率**: 完全相同的像素比例
7. **MSE 相似度**: 基於均方誤差的相似度

### 旋轉角度搜索策略

採用三階段優化策略：

- **第一階段（粗搜索）**: 每 3 度搜索，縮放比例 0.2，快速定位大致範圍
- **第二階段（完整評估）**: 在最佳候選角度範圍內，使用完整尺寸圖像評估
- **第三階段（細搜索）**: 在最佳角度 ±2 度範圍內，每 0.5 度搜索，偏移範圍 ±10 像素

總共約 49-100 次旋轉評估（根據候選數量動態調整），大幅提升處理速度。

### 平移校正

- 自動檢測圖像中心點偏移
- 支持手動調整印鑑位置
- 優化對齊算法，提升比對準確度

## 開發說明

### 資料庫遷移

使用 Alembic 進行資料庫遷移：

```bash
cd backend

# 創建遷移
alembic revision --autogenerate -m "描述"

# 應用遷移
alembic upgrade head

# 回滾遷移
alembic downgrade -1
```

### 查看日誌

```bash
# 查看所有服務日誌
docker-compose logs -f

# 查看特定服務日誌
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f postgres
```

### 重新構建

如果修改了代碼，需要重新構建：

```bash
docker-compose up -d --build
```

### 清理數據

```bash
# 停止並刪除容器（保留數據卷）
docker-compose down

# 停止並刪除容器和數據卷（**會刪除所有數據**）
docker-compose down -v
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

檢查資料庫連接字符串是否正確（在 `docker-compose.yml` 或 `backend/.env` 中）

### 3. 前端無法連接到後端

檢查：
- 後端是否正在運行（http://localhost:8000/health）
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

## 相關文檔

- [QUICKSTART.md](QUICKSTART.md) - 詳細的快速開始指南
- [VERIFY.md](VERIFY.md) - 驗證步驟指南
- [API 文檔](http://localhost:8000/docs) - 完整的 API 文檔（需要啟動服務）

## 授權

本專案僅供學習和研究使用。
