# 印鑑比對系統 - 前後端分離版本

這是一個現代化的印鑑比對系統，採用 FastAPI 後端 + React 前端的架構。

## 功能特點

### 核心功能

- **智能印鑑檢測**: 自動檢測圖像中的印鑑位置，支持手動調整
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

#### 圖像管理 API

- `POST /api/v1/images/upload` - 上傳圖像文件
- `GET /api/v1/images/{image_id}` - 獲取圖像信息
- `GET /api/v1/images/{image_id}/file` - 獲取圖像文件（返回實際圖像數據）
- `DELETE /api/v1/images/{image_id}` - 刪除圖像
- `POST /api/v1/images/{image_id}/detect-seal` - 檢測圖像中的印鑑位置
- `PUT /api/v1/images/{image_id}/seal-location` - 更新用戶確認的印鑑位置

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

## 專案結構

```
sealcompare/
├── backend/                           # 後端服務
│   ├── app/
│   │   ├── api/                      # API 路由
│   │   │   ├── images.py            # 圖像管理 API
│   │   │   ├── comparisons.py       # 比對管理 API
│   │   │   ├── visualizations.py    # 視覺化 API
│   │   │   └── statistics.py        # 統計 API
│   │   ├── services/                 # 業務邏輯層
│   │   │   ├── image_service.py     # 圖像服務
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
│   ├── Dockerfile                    # 後端 Docker 配置
│   ├── requirements.txt              # Python 依賴
│   └── alembic.ini                   # Alembic 遷移配置
├── frontend/                         # 前端應用
│   ├── src/
│   │   ├── components/              # React 組件
│   │   │   ├── ComparisonForm.jsx   # 比對表單
│   │   │   ├── ComparisonResult.jsx # 比對結果展示
│   │   │   ├── ImagePreview.jsx     # 圖像預覽
│   │   │   ├── SealDetectionBox.jsx # 印鑑檢測框
│   │   │   └── ...                  # 其他組件
│   │   ├── pages/                   # 頁面組件
│   │   │   ├── Home.jsx            # 首頁
│   │   │   ├── Comparison.jsx      # 比對頁面
│   │   │   └── History.jsx         # 歷史記錄頁面
│   │   ├── services/
│   │   │   └── api.js              # API 服務封裝
│   │   ├── App.jsx                  # 應用入口
│   │   └── main.jsx                 # React 入口
│   ├── Dockerfile                    # 前端 Docker 配置
│   ├── package.json                  # Node.js 依賴
│   └── vite.config.js                # Vite 配置
├── test_images/                      # 測試圖像文件
├── docker-compose.yml                # Docker Compose 編排配置
├── README.md                         # 專案說明文檔
├── QUICKSTART.md                     # 快速開始指南
├── VERIFY.md                         # 驗證步驟指南
├── main.py                           # CLI 工具入口（舊版）
└── seal_compare.py                   # CLI 工具核心（舊版）
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

- `DATABASE_URL`: PostgreSQL 連接字符串（預設：`postgresql://sealcompare:sealcompare@postgres:5432/sealcompare`）
- `REDIS_URL`: Redis 連接字符串（預設：`redis://redis:6379/0`）
- `UPLOAD_DIR`: 上傳文件存儲目錄（預設：`/app/uploads`）
- `LOGS_DIR`: 日誌和生成文件目錄（預設：`/app/logs`）
- `MAX_UPLOAD_SIZE`: 最大上傳文件大小（預設：10MB）
- `CORS_ORIGINS`: 允許的 CORS 來源（JSON 格式的列表）

### 比對算法說明

系統採用多種相似度指標融合算法，主要包括：

1. **SSIM (結構相似性指數)**: 衡量圖像結構相似度
2. **模板匹配**: 使用 OpenCV 模板匹配算法
3. **像素相似度**: 計算像素級差異
4. **直方圖相似度**: 比較圖像直方圖分佈
5. **邊緣相似度**: 自適應邊緣檢測和比較
6. **精確匹配率**: 完全相同的像素比例
7. **MSE 相似度**: 基於均方誤差的相似度

旋轉角度搜索採用三階段優化策略：
- **第一階段**: 每 15 度搜索（24 次），快速定位大致範圍
- **第二階段**: 在最佳角度 ±15 度範圍內，每 2 度搜索（16 次）
- **第三階段**: 在最佳角度 ±2 度範圍內，每 0.5 度搜索（9 次）

總共約 49 次旋轉評估，大幅提升處理速度。

## 授權

本專案僅供學習和研究使用。
