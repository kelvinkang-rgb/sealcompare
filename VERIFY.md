# 驗證步驟指南

## ✅ 步驟 1: 確認服務已啟動

所有服務應該都在運行中。如果看到以下狀態，表示成功：

```
NAME                   STATUS
sealcompare-backend    Up
sealcompare-frontend   Up  
sealcompare-postgres   Up (healthy)
sealcompare-redis      Up (healthy)
```

## ✅ 步驟 2: 驗證後端 API

### 2.1 健康檢查
打開瀏覽器訪問：
```
http://localhost:8000/health
```

應該看到：
```json
{"status": "healthy"}
```

### 2.2 API 文檔
訪問：
```
http://localhost:8000/docs
```

應該看到 Swagger UI 界面，可以測試所有 API。

### 2.3 根路徑
訪問：
```
http://localhost:8000/
```

應該看到 API 信息。

## ✅ 步驟 3: 驗證前端界面

打開瀏覽器訪問：
```
http://localhost:3000
```

應該看到首頁，包含：
- 標題「印鑑比對系統」
- 「開始比對」按鈕
- 「比對記錄」按鈕

## ✅ 步驟 4: 測試完整流程

### 4.1 使用前端界面測試

1. 訪問 http://localhost:3000
2. 點擊「開始比對」或導航到「圖像比對」頁面
3. 上傳兩個測試圖像：
   - 圖像1: 選擇 `test_images/seal_original_1.jpg`
   - 圖像2: 選擇 `test_images/seal_rotated_medium.jpg`
4. 設置閾值（預設 0.95）
5. 點擊「開始比對」
6. 等待處理完成（會顯示進度）
7. 查看結果：
   - 相似度百分比
   - 匹配/不匹配狀態
   - 並排對比圖
   - 差異熱力圖

### 4.2 使用 API 測試（可選）

#### 使用 Swagger UI（最簡單）

1. 訪問 http://localhost:8000/docs
2. 展開 `POST /api/v1/images/upload`
3. 點擊「Try it out」
4. 選擇一個測試圖像文件
5. 點擊「Execute」
6. 複製返回的 `id`

重複上傳第二個圖像，然後：

1. 展開 `POST /api/v1/comparisons/`
2. 點擊「Try it out」
3. 填入兩個圖像的 ID
4. 點擊「Execute」
5. 複製返回的 `comparison_id`

最後：

1. 展開 `GET /api/v1/comparisons/{comparison_id}`
2. 點擊「Try it out」
3. 填入 comparison_id
4. 點擊「Execute」
5. 查看比對結果

## ✅ 步驟 5: 查看比對記錄

1. 在前端點擊「比對記錄」
2. 或訪問 http://localhost:3000/history
3. 查看所有歷史比對結果

## 🔍 故障排查

### 如果服務沒有啟動

查看日誌：
```powershell
docker-compose logs -f
```

查看特定服務日誌：
```powershell
docker-compose logs backend
docker-compose logs frontend
docker-compose logs postgres
```

### 如果端口被占用

檢查端口使用情況：
```powershell
netstat -ano | findstr :8000
netstat -ano | findstr :3000
```

如果被占用，可以修改 `docker-compose.yml` 中的端口映射。

### 如果後端連接失敗

檢查資料庫是否正常：
```powershell
docker-compose ps postgres
```

重啟服務：
```powershell
docker-compose restart backend
```

### 如果前端無法連接後端

1. 確認後端正在運行（http://localhost:8000/health）
2. 檢查瀏覽器控制台是否有錯誤
3. 檢查 CORS 設置（在 `backend/app/config.py` 中）

## 📝 下一步

- 閱讀 `QUICKSTART.md` 了解更多使用細節
- 查看 `README.md` 了解完整功能
- 訪問 http://localhost:8000/docs 查看完整 API 文檔

