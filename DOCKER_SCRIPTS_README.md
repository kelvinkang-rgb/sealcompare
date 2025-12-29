# Docker Compose 管理腳本說明

本文檔說明如何使用提供的腳本來管理 docker-compose 服務。

## 通用腳本

### 1. 啟動服務

#### Windows (start.bat)
```batch
start.bat
```

#### Linux/macOS (start.sh)
```bash
./start.sh
```

功能：啟動所有服務（首次運行會自動構建）

---

### 2. 重啟服務（不重新構建）

當服務已運行，只是需要重啟時使用（速度快，適合驗證配置更改）。

#### Windows (restart.bat)
```batch
restart.bat
```

#### Linux/macOS (restart.sh)
```bash
./restart.sh
```

功能：重啟所有服務，不重新構建映像

---

### 3. 重新構建並啟動

當代碼更改後，需要重新構建映像時使用。

#### Windows (rebuild.bat)
```batch
rebuild.bat
```

#### Linux/macOS (rebuild.sh)
```bash
./rebuild.sh
```

功能：停止服務 → 重新構建映像 → 啟動服務

---

### 4. 停止服務

#### Windows (stop.bat)
```batch
stop.bat
```

#### Linux/macOS (stop.sh)
```bash
./stop.sh
```

功能：停止所有服務（容器仍在，可以快速重新啟動）

---

### 5. 查看日誌

#### Windows (logs.bat)
```batch
logs.bat              # 查看所有服務日誌
logs.bat backend      # 查看後端日誌
logs.bat frontend     # 查看前端日誌
logs.bat postgres     # 查看資料庫日誌
logs.bat redis        # 查看 Redis 日誌
```

#### Linux/macOS (logs.sh)
```bash
./logs.sh              # 查看所有服務日誌
./logs.sh backend      # 查看後端日誌
./logs.sh frontend     # 查看前端日誌
./logs.sh postgres     # 查看資料庫日誌
./logs.sh redis        # 查看 Redis 日誌
```

功能：實時查看服務日誌（按 Ctrl+C 退出）

---

### 6. 查看服務狀態

#### Windows (status.bat)
```batch
status.bat
```

#### Linux/macOS (status.sh)
```bash
./status.sh
```

功能：查看所有服務的運行狀態

---

### 7. 清理服務

#### Windows (clean.bat)
```batch
clean.bat
```

#### Linux/macOS (clean.sh)
```bash
./clean.sh
```

功能：停止並刪除所有容器和網絡（**但保留數據卷**，資料庫數據不會丟失）

**注意**：此操作會刪除容器，但不會刪除映像和數據卷。

---

## 單服務腳本

### 8. 重啟後端服務

用於後端代碼更改後的快速重啟（後端使用 volume 掛載，代碼更改會自動重新加載）。

#### Windows (restart-backend.bat)
```batch
restart-backend.bat
```

#### Linux/macOS (restart-backend.sh)
```bash
./restart-backend.sh
```

功能：只重啟後端服務（速度快）

---

### 9. 重新構建前端服務

用於前端代碼或配置更改後，需要重新構建前端映像時使用。

#### Windows (rebuild-frontend.bat)
```batch
rebuild-frontend.bat
```

#### Linux/macOS (rebuild-frontend.sh)
```bash
./rebuild-frontend.sh
```

功能：重新構建並啟動前端服務（前端更改需要重新構建）

---

## 部署腳本

### 10. 完整部署

#### Windows (deploy.bat)
```batch
deploy.bat
```

#### Linux/macOS (deploy.sh)
```bash
./deploy.sh
```

功能：停止服務 → 重新構建 → 啟動服務 → 顯示狀態

---

## 常見使用場景

### 場景1：首次啟動項目
```bash
./start.sh          # 或 start.bat (Windows)
```

### 場景2：修改後端代碼後驗證
```bash
./restart-backend.sh    # 後端使用 volume 掛載，重啟即可
# 或
./restart.sh            # 重啟所有服務
```

### 場景3：修改前端代碼後驗證
```bash
./rebuild-frontend.sh   # 前端需要重新構建
```

### 場景4：同時修改前後端代碼後驗證
```bash
./rebuild.sh            # 重新構建所有服務
```

### 場景5：查看後端錯誤日誌
```bash
./logs.sh backend       # 或 logs.bat backend (Windows)
```

### 場景6：檢查服務是否正常運行
```bash
./status.sh             # 或 status.bat (Windows)
```

### 場景7：快速重啟服務（無代碼更改）
```bash
./restart.sh            # 或 restart.bat (Windows)
```

### 場景8：清理環境（保留數據）
```bash
./clean.sh              # 或 clean.bat (Windows)
# 然後重新啟動
./start.sh
```

---

## 服務訪問地址

- **前端**: http://localhost:3000
- **後端 API**: http://localhost:8000
- **API 文檔**: http://localhost:8000/docs

---

## 注意事項

1. **首次運行**：首次運行 `start.sh` 或 `start.bat` 會自動構建映像，可能需要較長時間。

2. **前端更改**：修改前端代碼後，必須使用 `rebuild-frontend.sh` 重新構建，因為前端是靜態構建的。

3. **後端更改**：修改後端代碼後，可以使用 `restart-backend.sh` 快速重啟，因為後端使用 volume 掛載，代碼會自動重新加載。

4. **Docker Compose 版本**：腳本會自動檢測使用 `docker compose`（新版本）或 `docker-compose`（舊版本）。

5. **數據持久化**：使用 `clean.sh` 清理服務不會刪除數據卷，資料庫數據會保留。如需完全清理，請手動執行：
   ```bash
   docker compose down -v  # 刪除包括數據卷在內的所有資源
   ```

---

## 腳本列表

### Windows 腳本 (.bat)
- `start.bat` - 啟動服務
- `restart.bat` - 重啟服務
- `rebuild.bat` - 重新構建並啟動
- `stop.bat` - 停止服務
- `logs.bat` - 查看日誌
- `status.bat` - 查看狀態
- `clean.bat` - 清理服務
- `restart-backend.bat` - 重啟後端
- `rebuild-frontend.bat` - 重新構建前端
- `deploy.bat` - 完整部署

### Linux/macOS 腳本 (.sh)
- `start.sh` - 啟動服務
- `restart.sh` - 重啟服務
- `rebuild.sh` - 重新構建並啟動
- `stop.sh` - 停止服務
- `logs.sh` - 查看日誌
- `status.sh` - 查看狀態
- `clean.sh` - 清理服務
- `restart-backend.sh` - 重啟後端
- `rebuild-frontend.sh` - 重新構建前端
- `deploy.sh` - 完整部署

