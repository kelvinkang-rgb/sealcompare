# 部署腳本使用說明

## 快速開始

### Linux/macOS 用戶

```bash
# 初始化部署（首次使用）
./deploy.sh init

# 快速更新（程式碼修改後）
./deploy.sh deploy

# 查看狀態
./deploy.sh status
```

### Windows 用戶

**方式 1：使用 Git Bash（推薦）**
```bash
bash deploy.sh init
bash deploy.sh deploy
bash deploy.sh status
```

**方式 2：使用 deploy.bat**
```cmd
.\deploy.bat
```

**方式 3：使用 WSL**
```bash
wsl bash deploy.sh init
```

## 所有命令

### 互動式選單
不帶參數執行，顯示選單供選擇：
```bash
./deploy.sh
```

### 命令參數模式

```bash
./deploy.sh init      # 初始化部署（首次使用，含資料庫遷移）
./deploy.sh deploy    # 快速更新（僅重啟服務，不重建）
./deploy.sh rebuild   # 完全重建（重建 Docker images，保留資料）
./deploy.sh restart   # 重啟所有服務
./deploy.sh stop      # 停止所有服務
./deploy.sh status    # 查看服務狀態
./deploy.sh --help    # 顯示幫助訊息
```

## 命令說明

### init - 初始化部署
- **用途**：首次部署系統
- **動作**：
  - 構建所有 Docker images
  - 啟動所有服務（postgres, redis, backend, frontend）
  - 等待健康檢查通過
  - 執行資料庫遷移
- **耗時**：約 3-5 分鐘（首次構建）
- **使用時機**：
  - 首次安裝系統
  - 長時間停止後重新啟動

### deploy - 快速更新
- **用途**：更新程式碼後快速部署
- **動作**：
  - 僅重啟 backend 和 frontend 容器
  - 不重建 Docker images
- **耗時**：約 10-20 秒
- **使用時機**：
  - Python 程式碼修改
  - React 程式碼修改
  - 配置檔案修改

### rebuild - 完全重建
- **用途**：重建所有 Docker images
- **動作**：
  - 停止所有服務
  - 重建所有 images
  - 啟動服務
  - **保留資料庫和上傳檔案**
- **耗時**：約 2-5 分鐘
- **使用時機**：
  - Dockerfile 修改
  - requirements.txt 修改
  - package.json 修改
  - 系統依賴變更

### restart - 重啟服務
- **用途**：重啟所有容器
- **動作**：
  - 重啟所有容器（不重建）
- **耗時**：約 10-20 秒
- **使用時機**：
  - 環境變數修改
  - docker-compose.yml 配置修改

### stop - 停止服務
- **用途**：停止所有服務
- **動作**：
  - 優雅停止所有容器
  - 保留 volumes 和 images
- **使用時機**：
  - 暫時不使用系統
  - 系統維護

### status - 查看狀態
- **用途**：查看服務狀態
- **顯示**：
  - 容器運行狀態
  - 資源使用情況（CPU、記憶體）
  - 服務訪問地址

## 常見問題

### Q: deploy 和 rebuild 的區別？
**A:** 
- `deploy`：僅重啟容器，不重建 images（快速，適合程式碼修改）
- `rebuild`：重建 Docker images（慢，適合依賴變更）

### Q: rebuild 會刪除資料嗎？
**A:** 不會。rebuild 會保留：
- 資料庫資料（postgres_data volume）
- 上傳檔案（uploads_data volume）
- 日誌檔案（logs_data volume）

### Q: 如何完全清理系統？
**A:** 使用以下命令（⚠️ 會刪除所有資料）：
```bash
docker compose down -v
```

### Q: 腳本執行失敗怎麼辦？
**A:** 檢查以下項目：
1. Docker 是否安裝並運行？
   ```bash
   docker --version
   docker info
   ```
2. Docker Compose 是否安裝？
   ```bash
   docker compose version
   ```
3. 是否在專案根目錄執行？
   ```bash
   ls docker-compose.yml  # 應該存在
   ```
4. 查看詳細錯誤：
   ```bash
   docker compose logs
   ```

### Q: 服務啟動後無法訪問？
**A:** 檢查：
1. 容器是否都在運行？
   ```bash
   ./deploy.sh status
   ```
2. 端口是否被佔用？
   - Frontend: http://localhost:3000
   - Backend: http://localhost:8000
3. 防火牆是否阻擋？

## 服務訪問

部署成功後，可以訪問：

- **前端界面**: http://localhost:3000
- **後端 API**: http://localhost:8000
- **API 文檔**: http://localhost:8000/docs

## 日誌查看

```bash
# 查看所有服務日誌
docker compose logs -f

# 查看特定服務日誌
docker compose logs -f backend
docker compose logs -f frontend

# 查看最近 100 行日誌
docker compose logs --tail=100
```

## 進階操作

### 進入容器內部
```bash
# 進入 backend 容器
docker compose exec backend bash

# 進入 postgres 容器
docker compose exec postgres psql -U sealcompare
```

### 手動執行資料庫遷移
```bash
docker compose exec backend alembic upgrade head
```

### 查看資源使用
```bash
docker stats
```

## 舊版腳本

如果需要使用舊版的簡單部署腳本，已備份為：
- `deploy.sh.backup`

可以恢復使用：
```bash
cp deploy.sh.backup deploy.sh
```

