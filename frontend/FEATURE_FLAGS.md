# 功能開關配置文檔

本文檔說明如何使用環境變數控制前端組件的顯示/隱藏。

## 概述

通過 Docker 構建參數（`ARG`）傳遞功能開關，在構建時注入到前端應用中。使用扁平化命名的環境變數格式：`VITE_FEATURE_<COMPONENT_NAME>`。

## 可用功能開關

| 環境變數 | 組件 | 描述 | 預設值 |
|---------|------|------|--------|
| `VITE_FEATURE_BATCH_SEAL_ADJUSTMENT` | BatchSealAdjustment | 批量調整印鑑位置組件 | `true` |
| `VITE_FEATURE_METRICS_EXPLANATION_DIALOG` | MetricsExplanationDialog | 比對指標說明對話框 | `true` |
| `VITE_FEATURE_SIMILARITY_HISTOGRAM` | SimilarityHistogram | 相似度直方圖組件 | `true` |
| `VITE_FEATURE_COMPARISON_EDIT_DIALOG` | ComparisonEditDialog | 比對記錄編輯對話框 | `true` |
| `VITE_FEATURE_DELETE_CONFIRM_DIALOG` | DeleteConfirmDialog | 刪除確認對話框 | `true` |
| `VITE_FEATURE_PROCESSING_STAGES` | ProcessingStages | 處理階段顯示組件 | `true` |
| `VITE_FEATURE_VERIFICATION_VIEW` | VerificationView | 校正驗證視圖組件 | `true` |
| `VITE_FEATURE_IMAGE_MODAL` | ImageModal | 圖像模態框組件 | `true` |
| `VITE_FEATURE_IMAGE_PREVIEW_DIALOG` | ImagePreviewDialog | 圖像預覽對話框組件 | `true` |
| `VITE_FEATURE_MASK_STATISTICS` | Mask統計資訊 | 顯示 mask 統計資訊區域 | `true` |
| `VITE_FEATURE_TIMING_DETAILS` | 時間詳情 | 顯示每個比對結果的時間詳情 | `true` |
| `VITE_FEATURE_TASK_TIMING_STATISTICS` | 任務時間統計 | 顯示任務級別的時間統計 | `true` |
| `VITE_FEATURE_ADVANCED_SETTINGS` | 進階設定 | 顯示進階設定 Accordion（包含所有進階設定） | `true` |
| `VITE_FEATURE_MAX_SEALS_SETTING` | 比對印鑑數量上限設定 | 在進階設定中的印鑑數量上限設定 | `true` |
| `VITE_FEATURE_THRESHOLD_SETTING` | 相似度閾值設定 | 相似度閾值設定（MultiSealTest 和 Comparison 頁面） | `true` |
| `VITE_FEATURE_MASK_WEIGHTS_SETTING` | Mask相似度權重參數設定 | 在進階設定中的 mask 相似度權重參數設定 | `true` |
| `VITE_FEATURE_ALIGNMENT_TIMING_DETAILS` | 對齊時間詳情 | 顯示對齊過程的時間詳情（ComparisonResult） | `true` |

## 配置方式

### 方式1：在 docker-compose.yml 中配置（推薦）

```yaml
frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile
    args:
      VITE_API_BASE_URL: /api/v1
      # 功能開關配置
      VITE_FEATURE_METRICS_EXPLANATION_DIALOG: "false"
      VITE_FEATURE_SIMILARITY_HISTOGRAM: "false"
```

### 方式2：使用 .env 文件

創建 `.env` 文件：

```bash
VITE_FEATURE_METRICS_EXPLANATION_DIALOG=false
VITE_FEATURE_SIMILARITY_HISTOGRAM=false
```

然後在 `docker-compose.yml` 中引用：

```yaml
frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile
    args:
      VITE_FEATURE_METRICS_EXPLANATION_DIALOG: ${VITE_FEATURE_METRICS_EXPLANATION_DIALOG:-"true"}
      VITE_FEATURE_SIMILARITY_HISTOGRAM: ${VITE_FEATURE_SIMILARITY_HISTOGRAM:-"true"}
```

### 方式3：命令行構建時指定

```bash
docker compose build frontend \
  --build-arg VITE_FEATURE_METRICS_EXPLANATION_DIALOG=false \
  --build-arg VITE_FEATURE_SIMILARITY_HISTOGRAM=false
```

## 使用範例

### 範例1：隱藏指標說明對話框和相似度直方圖

```yaml
# docker-compose.yml
frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile
    args:
      VITE_FEATURE_METRICS_EXPLANATION_DIALOG: "false"
      VITE_FEATURE_SIMILARITY_HISTOGRAM: "false"
```

### 範例2：隱藏批量調整功能

```yaml
frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile
    args:
      VITE_FEATURE_BATCH_SEAL_ADJUSTMENT: "false"
```

### 範例3：隱藏歷史記錄頁面的編輯和刪除功能

```yaml
frontend:
  build:
    context: ./frontend
    dockerfile: Dockerfile
    args:
      VITE_FEATURE_COMPARISON_EDIT_DIALOG: "false"
      VITE_FEATURE_DELETE_CONFIRM_DIALOG: "false"
```

## 注意事項

1. **構建時注入**：功能開關是在構建時注入的，修改配置後需要重新構建 Docker 映像才能生效。

2. **預設行為**：所有功能預設啟用（`true`），未設置環境變數時保持可見，確保向後兼容。

3. **字符串格式**：環境變數的值必須是字符串格式（`"true"` 或 `"false"`），注意引號的使用。

4. **重新構建**：修改功能開關後，執行以下命令重新構建：
   ```bash
   docker compose build frontend
   docker compose up -d
   ```

## 技術實現

功能開關通過以下方式實現：

1. **配置文件**：`frontend/src/config/featureFlags.js`
   - 定義所有功能開關常數
   - 從環境變數讀取配置
   - 提供 `useFeatureFlag` Hook 和 `isFeatureEnabled` 函數

2. **Dockerfile**：`frontend/Dockerfile`
   - 使用 `ARG` 接收構建參數
   - 通過 `ENV` 傳遞給構建過程

3. **組件使用**：
   ```javascript
   import { useFeatureFlag, FEATURE_FLAGS } from '../config/featureFlags'
   
   function MyComponent() {
     const showFeature = useFeatureFlag(FEATURE_FLAGS.FEATURE_NAME)
     
     return (
       <div>
         {showFeature && <FeatureComponent />}
       </div>
     )
   }
   ```

## 添加新功能開關

如需添加新的功能開關：

1. 在 `frontend/src/config/featureFlags.js` 中添加常數定義
2. 在 `frontend/Dockerfile` 中添加對應的 `ARG` 和 `ENV`
3. 在 `docker-compose.yml` 中添加參數（可選，使用預設值時可不添加）
4. 在組件中使用 `useFeatureFlag` Hook 控制顯示
5. 更新此文檔

## 相關文件

- 配置文件：`frontend/src/config/featureFlags.js`
- Dockerfile：`frontend/Dockerfile`
- 配置範例：`frontend/.env.example`

