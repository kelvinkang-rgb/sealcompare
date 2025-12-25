# Debug 指南

本文件說明如何調試多印鑑比對系統中的問題。

## 查看後端日誌

### 基本命令

```bash
# 查看最近的日誌（最後100行）
docker-compose logs --tail=100 backend

# 實時查看日誌（跟隨模式）
docker-compose logs -f backend

# 查看特定時間範圍的日誌
docker-compose logs --since 10m backend  # 最近10分鐘
docker-compose logs --since 1h backend   # 最近1小時

# 查看錯誤日誌
docker-compose logs --tail=100 backend | grep -i error

# 查看警告日誌
docker-compose logs --tail=100 backend | grep -i warning
```

### 查看特定功能的日誌

```bash
# 查看多印鑑比對相關日誌
docker-compose logs --tail=200 backend | grep -i "多印鑑比對\|服務層"

# 查看圖像處理相關日誌
docker-compose logs --tail=200 backend | grep -i "圖像\|image"

# 查看線程相關日誌
docker-compose logs --tail=200 backend | grep -i "線程\|thread"
```

## 前端錯誤調試

### 瀏覽器 Console

1. 打開瀏覽器開發者工具（F12）
2. 切換到 Console 標籤
3. 當比對失敗時，查看以下日誌：
   - `比對失敗 - 完整錯誤信息:` - 包含 HTTP 狀態碼、錯誤詳情等
   - `比對失敗 - 錯誤對象:` - 完整的錯誤對象
   - `比對失敗 - 錯誤堆疊:` - 錯誤堆疊信息

### 錯誤詳情對話框

當比對失敗時，前端會顯示錯誤訊息，點擊「查看詳情」按鈕可以：

1. 查看完整的錯誤信息
2. 查看 HTTP 狀態碼
3. 查看發生時間
4. 查看操作參數（印鑑數量、圖像ID等）
5. 展開查看完整錯誤對象（JSON 格式）
6. 複製錯誤信息到剪貼板

### Network 標籤

1. 打開瀏覽器開發者工具（F12）
2. 切換到 Network 標籤
3. 發起比對請求
4. 查看請求詳情：
   - 請求 URL
   - 請求方法
   - 請求參數
   - 響應狀態碼
   - 響應內容

## 常見錯誤和解決方法

### 1. 比對流程失敗 - 保存階段

**錯誤訊息**: `保存印鑑位置失敗`

**可能原因**:
- 圖像2未正確上傳
- 印鑑數據格式錯誤
- 資料庫連接問題

**解決方法**:
1. 檢查後端日誌：`docker-compose logs --tail=50 backend | grep -i "保存\|save"`
2. 確認圖像2已正確上傳
3. 檢查印鑑數據是否有效
4. 確認資料庫服務正常運行：`docker-compose ps postgres`

### 2. 比對流程失敗 - 裁切階段

**錯誤訊息**: `裁切印鑑失敗`

**可能原因**:
- 印鑑位置數據無效
- 圖像文件損壞
- 磁碟空間不足

**解決方法**:
1. 檢查後端日誌：`docker-compose logs --tail=50 backend | grep -i "裁切\|crop"`
2. 確認印鑑位置數據正確
3. 檢查圖像文件是否存在：`docker-compose exec backend ls -lh /app/uploads/`
4. 檢查磁碟空間：`docker-compose exec backend df -h`

### 3. 比對流程失敗 - 比對階段

**錯誤訊息**: `比對失敗`

**可能原因**:
- 圖像處理失敗
- 線程池配置問題
- 記憶體不足
- 超時

**解決方法**:
1. 檢查後端日誌：`docker-compose logs --tail=100 backend | grep -i "比對\|compare\|服務層"`
2. 查看具體的錯誤堆疊信息
3. 檢查線程數配置：`backend/app/config.py` 中的 `MAX_COMPARISON_THREADS`
4. 檢查容器資源使用：`docker stats`
5. 如果是超時問題，檢查前端超時設置：`frontend/src/services/api.js` 中的 `compareImage1WithSeals` 函數

### 4. HTTP 500 錯誤

**錯誤訊息**: `Internal Server Error`

**可能原因**:
- 後端未預期的異常
- 依賴服務問題（資料庫、Redis）

**解決方法**:
1. 查看完整的後端日誌：`docker-compose logs --tail=200 backend`
2. 查看錯誤堆疊信息
3. 檢查依賴服務狀態：`docker-compose ps`
4. 重啟服務：`docker-compose restart backend`

### 5. 超時錯誤

**錯誤訊息**: `timeout` 或 `Request timeout`

**可能原因**:
- 比對時間過長（印鑑數量太多）
- 網路問題
- 前端超時設置過短

**解決方法**:
1. 檢查前端超時設置（根據印鑑數量動態計算）
2. 減少同時比對的印鑑數量
3. 增加前端超時時間（在 `frontend/src/services/api.js` 中）
4. 檢查後端處理時間：查看日誌中的耗時信息

### 6. 結果缺失

**錯誤訊息**: `比對結果缺失（線程執行可能失敗）`

**可能原因**:
- 線程執行失敗
- 線程池配置問題
- 資源競爭

**解決方法**:
1. 查看後端日誌中的警告信息：`docker-compose logs --tail=100 backend | grep -i "警告\|warning\|缺失"`
2. 檢查線程池配置
3. 減少並行線程數
4. 檢查系統資源使用情況

## 性能調試

### 查看比對耗時

後端日誌會記錄每個階段的耗時：

```
[服務層] 比對階段耗時: XX.XX 秒
[服務層] 總服務時間: XX.XX 秒
[服務層] 平均每個印鑑比對時間: XX.XX 秒
[服務層] 理論加速比: X.XXx
```

### 查看每個印鑑的處理時間

```
[服務層] 印鑑 X 比對完成，耗時: XX.XX 秒，相似度: X.XXXX
```

### 優化建議

1. **減少線程數**: 如果系統資源有限，減少 `MAX_COMPARISON_THREADS`
2. **增加超時時間**: 如果比對時間較長，增加前端超時設置
3. **分批處理**: 如果印鑑數量很多，考慮分批處理

## 日誌級別

後端使用 Python `logging` 模組，日誌級別包括：

- `INFO`: 一般信息（比對開始、完成、統計等）
- `WARNING`: 警告信息（結果數量不匹配等）
- `ERROR`: 錯誤信息（比對失敗、異常等）

## 獲取幫助

如果問題仍然無法解決：

1. 收集完整的錯誤信息：
   - 前端 Console 日誌
   - 前端 Network 請求詳情
   - 後端日誌（使用 `docker-compose logs --tail=500 backend > backend.log`）
   - 錯誤詳情對話框中的完整錯誤對象

2. 檢查系統狀態：
   - `docker-compose ps` - 查看服務狀態
   - `docker stats` - 查看資源使用
   - `docker-compose logs` - 查看所有服務日誌

3. 提供以下信息：
   - 錯誤發生的具體步驟
   - 錯誤訊息和堆疊信息
   - 系統配置（線程數、超時設置等）
   - 印鑑數量
   - 圖像大小和格式

