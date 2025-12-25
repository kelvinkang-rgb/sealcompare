# 多印鑑比對流程說明

## 概述

本文檔詳細說明多印鑑比對系統的完整流程，特別關注**結果呈現的處理**與 **Input、Output、Process** 的關係。

---

## 整體流程架構

```
前端 (MultiSealTest.jsx)
    ↓
API 層 (images.py)
    ↓
服務層 (image_service.py)
    ↓
核心比對 (seal_compare.py)
    ↓
結果呈現 (MultiSealComparisonResults.jsx)
```

---

## 階段一：輸入準備 (Input Preparation)

### 1.1 前端輸入

**Input:**
- `image1`: 圖像1（單印鑑圖像）
- `image2`: 圖像2（包含多個印鑑的圖像）
- `maxSeals`: 最大檢測印鑑數量（預設 6）
- `threshold`: 相似度閾值（預設 0.5）
- `similaritySsimWeight`: SSIM 權重（預設 0.5）
- `similarityTemplateWeight`: Template Match 權重（預設 0.35）
- `pixelSimilarityWeight`: Pixel Similarity 權重（預設 0.1）
- `histogramSimilarityWeight`: Histogram Similarity 權重（預設 0.05）

**Process:**
1. 上傳圖像1 → 自動檢測單印鑑位置 → 保存 `seal_bbox`
2. 上傳圖像2 → 自動檢測多印鑑位置 → 保存 `multiple_seals` 陣列
3. 用戶確認/調整印鑑位置
4. 裁切印鑑 → 生成多個裁切後的圖像（`croppedImageIds`）

**Output:**
- `image1.seal_bbox`: 圖像1的印鑑邊界框
- `image2.multiple_seals`: 圖像2的多印鑑位置陣列
- `croppedImageIds`: 裁切後的印鑑圖像 ID 列表

**關鍵代碼位置:**
```583:592:frontend/src/pages/MultiSealTest.jsx
    compareMutation.mutate({
      image1Id: uploadImage1Mutation.data.id,
      sealImageIds: croppedImageIds,
      threshold: threshold,
      similaritySsimWeight: similaritySsimWeight,
      similarityTemplateWeight: similarityTemplateWeight,
      pixelSimilarityWeight: pixelSimilarityWeight,
      histogramSimilarityWeight: histogramSimilarityWeight
    })
```

---

## 階段二：比對任務創建 (Task Creation)

### 2.1 API 層處理

**Input:**
- `image1_id`: 圖像1 ID
- `seal_image_ids`: 裁切後的印鑑圖像 ID 列表
- `threshold`: 相似度閾值
- 各種相似度權重參數

**Process:**
1. 生成任務 UID (`task_uid`)
2. 創建 `MultiSealComparisonTask` 記錄
3. 設置狀態為 `PENDING`
4. 將任務加入後台處理隊列

**Output:**
- `task_uid`: 任務唯一標識符
- `task_id`: 資料庫記錄 ID
- 任務狀態: `pending`

**關鍵代碼位置:**
```283:322:backend/app/api/images.py
@router.post("/{image1_id}/compare-with-seals", response_model=MultiSealComparisonTaskResponse)
def compare_image1_with_seals(
    image1_id: UUID,
    request: MultiSealComparisonRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    # 生成任務 UID
    task_uid = str(uuid_lib.uuid4())
    
    # 創建任務記錄
    task = MultiSealComparisonTask(
        task_uid=task_uid,
        image1_id=image1_id,
        status=ComparisonStatus.PENDING,
        seal_image_ids=[str(sid) for sid in request.seal_image_ids],
        threshold=request.threshold,
        ...
    )
```

---

## 階段三：後台比對處理 (Background Processing)

### 3.1 任務初始化

**Input:**
- `task_uid`: 任務 UID
- `image1_id`: 圖像1 ID
- `seal_image_ids`: 印鑑圖像 ID 列表

**Process:**
1. 更新任務狀態為 `PROCESSING`
2. 初始化結果列表（`results = []`）
3. 設置總數（`total_count = len(seal_image_ids)`）

**Output:**
- 任務狀態: `processing`

**關鍵代碼位置:**
```337:356:backend/app/api/images.py
            # 更新狀態為處理中
            task_record.status = ComparisonStatus.PROCESSING
            task_record.started_at = datetime.utcnow()
            task_record.results = []  # 初始化結果列表
            task_record.total_count = len(seal_image_ids)
```

### 3.2 並行比對處理

**Input:**
- `image1_cropped_path`: 裁切後的圖像1路徑
- `seal_image_ids`: 印鑑圖像 ID 列表
- 比對參數（閾值、權重等）

**Process:**
1. **預載入所有印鑑圖像資訊**（線程安全）
2. **創建線程池**（`ThreadPoolExecutor`）
3. **並行處理每個印鑑**：
   - 調用 `_compare_single_seal()` 處理單個印鑑
   - 每個印鑑比對完成後，立即調用 `task_update_callback()` 更新任務記錄
4. **等待所有任務完成**

**Output (每個印鑑):**
- `seal_index`: 印鑑索引（1, 2, 3, ...）
- `seal_image_id`: 印鑑圖像 ID
- `similarity`: 相似度分數（0-1）
- `is_match`: 是否匹配（boolean）
- `overlay1_path`: 疊圖1路徑（圖像1疊在印鑑上）
- `overlay2_path`: 疊圖2路徑（印鑑疊在圖像1上）
- `heatmap_path`: 差異熱力圖路徑
- `input_image1_path`: 輸入圖像1路徑（去背景後的圖像1）
- `input_image2_path`: 輸入圖像2路徑（對齊後的印鑑圖像）
- `error`: 錯誤訊息（如果有）

**關鍵代碼位置:**
```560:615:backend/app/services/image_service.py
            def process_seal(seal_data):
                """處理單個印鑑的比對（線程函數）"""
                idx, seal_image_id = seal_data
                try:
                    result = self._compare_single_seal(...)
                    # 確保結果被添加到列表
                    with results_lock:
                        results.append(result)
                    
                    # 如果有回調函數，立即更新任務記錄
                    if task_update_callback and result:
                        task_update_callback(result, idx + 1, len(seal_image_ids))
```

### 3.3 即時結果更新（回調機制）

**Input:**
- `result`: 單個印鑑的比對結果
- `current_index`: 當前印鑑索引
- `total_count`: 總印鑑數量

**Process:**
1. **線程安全更新**（使用 `with_for_update()` 鎖定記錄）
2. **合併結果**（使用字典追蹤，以 `seal_index` 為鍵）
3. **更新成功數量**：`success_count = sum(1 for r in results if r.get('error') is None)`
4. **提交到資料庫**

**Output:**
- `task.results`: 更新後的結果列表（按 `seal_index` 排序）
- `task.success_count`: 成功比對的數量

**關鍵代碼位置:**
```359:416:backend/app/api/images.py
            def update_task_with_result(result: Dict, current_index: int, total_count: int):
                """回調函數：當每個印鑑比對完成時更新任務記錄（線程安全）"""
                # 使用 with_for_update() 鎖定任務記錄
                task = db_task.query(MultiSealComparisonTask).filter(
                    MultiSealComparisonTask.task_uid == task_uid_str
                ).with_for_update().first()
                
                # 轉換結果為 JSON 格式
                result_json = {...}
                
                # 使用字典追蹤已存在的結果（以 seal_index 為鍵）
                results_dict = {r.get('seal_index'): r for r in current_results}
                results_dict[result['seal_index']] = result_json
                
                # 轉換回列表並按 seal_index 排序
                current_results = [results_dict[key] for key in sorted(results_dict.keys())]
                
                        # 計算成功數量
                        completed_count = len(current_results)
                        success_count = sum(1 for r in current_results if r.get('error') is None)
                        
                        # 更新任務記錄
                        task.results = current_results
                        task.success_count = success_count
```

### 3.4 單個印鑑比對處理 (`_compare_single_seal`)

**Input:**
- `image1_cropped_path`: 裁切後的圖像1路徑
- `seal_image_path`: 印鑑圖像路徑
- `seal_index`: 印鑑索引
- 比對參數

**Process:**
1. **載入圖像**
2. **圖像1處理**：
   - 去背景（`_auto_detect_bounds_and_remove_background`）
   - 保存為 `input_image1_path`
3. **圖像2處理**：
   - 去背景
   - 對齊優化（`_align_image2_to_image1`）
   - 保存為 `input_image2_path`
4. **相似度計算**：
   - SSIM 相似度
   - Template Match 相似度
   - Pixel Similarity
   - Histogram Similarity
   - 加權組合計算最終相似度
5. **生成視覺化**：
   - 疊圖1（`overlay1_path`）
   - 疊圖2（`overlay2_path`）
   - 差異熱力圖（`heatmap_path`）

**Output:**
- 完整的比對結果字典（包含所有路徑和相似度資訊）

---

## 階段四：結果驗證與完成 (Result Validation & Completion)

### 4.1 結果驗證

**Input:**
- `task.results`: 比對結果列表
- `expected_count`: 預期結果數量

**Process:**
1. **驗證結果數量**：檢查是否所有印鑑都有結果
2. **檢查缺失的印鑑**：識別缺失的 `seal_index`
3. **重新比對缺失的印鑑**（如果有的話）
4. **最終驗證**：確保結果數量正確

**Output:**
- 完整的結果列表（所有印鑑都有結果）

**關鍵代碼位置:**
```446:579:backend/app/api/images.py
            # 最終更新任務為完成前，驗證結果數量
            if task_record:
                current_results = task_record.results or []
                expected_count = len(seal_image_ids)
                actual_count = len(current_results)
                
                # 驗證結果數量是否正確
                if actual_count != expected_count:
                    # 檢查缺失的印鑑索引
                    # 重新比對缺失的印鑑
                    # ...
                
                # 更新任務為完成
                task_record.status = ComparisonStatus.COMPLETED
```

### 4.2 任務完成

**Input:**
- 驗證後的完整結果列表

**Process:**
1. 更新任務狀態為 `COMPLETED`
2. 記錄完成時間
3. 計算最終統計資訊

**Output:**
- 任務狀態: `completed`
- 最終結果列表

---

## 階段五：前端輪詢與結果呈現 (Frontend Polling & Result Display)

### 5.1 輪詢機制

**Input:**
- `currentTaskUid`: 當前任務 UID

**Process:**
1. **使用 React Query** 輪詢任務狀態
2. **輪詢頻率**: 每 1.5 秒一次
3. **停止條件**: 任務完成或失敗
4. **即時更新**: 每次輪詢都更新狀態和結果顯示

**Output:**
- `polledTaskResult`: 輪詢到的任務結果
- `taskStatus`: 任務狀態資訊
- `comparisonResults`: 比對結果列表

**關鍵代碼位置:**
```240:314:frontend/src/pages/MultiSealTest.jsx
  // 輪詢任務結果（包括狀態和部分結果）- 合併為單一輪詢
  const { data: polledTaskResult } = useQuery({
    queryKey: ['task-result', currentTaskUid],
    queryFn: () => imageAPI.getTaskResult(currentTaskUid),
    enabled: !!currentTaskUid && taskStatus?.status !== 'completed' && taskStatus?.status !== 'failed',
    refetchInterval: (query) => {
      const data = query.state.data
      // 如果任務完成或失敗，停止輪詢
      if (data?.status === 'completed' || data?.status === 'failed') {
        return false
      }
      // 否則每 1.5 秒輪詢一次
      return 1500
    },
  })
  
  // 當輪詢結果更新時，同時更新狀態和結果顯示
  useEffect(() => {
    if (polledTaskResult) {
      // 更新任務狀態
      setTaskStatus(statusInfo)
      
      // 處理結果顯示（即時顯示部分結果）
      if (polledTaskResult.results) {
        // 過濾出已完成疊圖的結果（有 overlay1_path 或 overlay2_path 的結果）
        const completedResults = polledTaskResult.results.filter(result => {
          return (result.overlay1_path || result.overlay2_path) || result.error
        })
        
        if (completedResults.length > 0) {
          // 確保結果按 seal_index 排序
          const sortedResults = [...completedResults].sort((a, b) => {
            const indexA = a.seal_index || 0
            const indexB = b.seal_index || 0
            return indexA - indexB
          })
          setComparisonResults(sortedResults)
        }
      }
    }
  }, [polledTaskResult, currentTaskUid])
```

### 5.2 結果呈現組件

**Input:**
- `results`: 比對結果列表（每個元素包含一個印鑑的比對結果）
- `image1Id`: 圖像1 ID（用於構建圖片 URL）

**Process:**
1. **構建圖片 URL**：
   - 從 `overlay1_path`、`overlay2_path`、`heatmap_path` 等路徑提取文件名
   - 構建完整的 API URL：`${API_BASE_URL}/images/multi-seal-comparisons/${fileName}`
2. **渲染結果卡片**：
   - 顯示印鑑索引
   - 顯示匹配狀態（匹配/不匹配）
   - 顯示相似度百分比
   - 顯示錯誤訊息（如果有）
3. **顯示視覺化圖片**：
   - 輸入圖像1（去背景後的圖像1）
   - 輸入圖像2（對齊後的印鑑圖像）
   - 疊圖1（圖像1疊在印鑑上）
   - 疊圖2（印鑑疊在圖像1上）
   - 差異熱力圖
4. **圖片點擊處理**：
   - 點擊圖片打開預覽對話框
   - 顯示完整尺寸的圖片

**Output:**
- 渲染的結果卡片列表
- 可點擊的圖片預覽

**關鍵代碼位置:**
```16:358:frontend/src/components/MultiSealComparisonResults.jsx
function MultiSealComparisonResults({ results, image1Id }) {
  // 構建圖片 URL
  const getImageUrl = (imagePath) => {
    if (!imagePath) return null
    const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ...
    const fileName = imagePath.split('/').pop()
    return `${API_BASE_URL}/images/multi-seal-comparisons/${fileName}`
  }
  
  // 渲染結果卡片
  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h6" gutterBottom>
        比對結果 ({results.length} 個印鑑)
      </Typography>
      
      <Grid container spacing={2}>
        {results.map((result, index) => (
          <Grid item xs={12} key={result.seal_image_id || index}>
            <Card>
              <CardContent>
                {/* 顯示印鑑索引和匹配狀態 */}
                {/* 顯示視覺化圖片 */}
                {/* 處理圖片點擊 */}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  )
}
```

---

## Input、Output、Process 關係圖

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT 階段                              │
├─────────────────────────────────────────────────────────────┤
│ • image1 (單印鑑圖像)                                        │
│ • image2 (多印鑑圖像)                                        │
│ • maxSeals, threshold, 權重參數                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    PROCESS 階段                              │
├─────────────────────────────────────────────────────────────┤
│ 1. 圖像上傳與檢測                                            │
│    → 檢測印鑑位置，保存 bbox                                 │
│                                                              │
│ 2. 印鑑裁切                                                  │
│    → 生成多個裁切後的圖像                                    │
│                                                              │
│ 3. 比對任務創建                                              │
│    → 生成 task_uid，創建任務記錄                             │
│                                                              │
│ 4. 並行比對處理（核心）                                      │
│    → 每個印鑑：去背景、對齊、相似度計算、生成視覺化          │
│    → 即時更新任務記錄（回調機制）                            │
│                                                              │
│ 5. 結果驗證與完成                                            │
│    → 驗證結果完整性，重新比對缺失的印鑑                      │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUT 階段                              │
├─────────────────────────────────────────────────────────────┤
│ 任務層輸出：                                                 │
│ • task_uid: 任務唯一標識符                                   │
│ • status: pending → processing → completed                  │
│ • results: 比對結果列表（每個印鑑一個結果）                  │
│   - seal_index: 印鑑索引                                    │
│   - similarity: 相似度分數                                   │
│   - is_match: 是否匹配                                       │
│   - overlay1_path, overlay2_path, heatmap_path             │
│   - input_image1_path, input_image2_path                    │
│   - error: 錯誤訊息（如果有）                                │
│                                                              │
│ 前端呈現輸出：                                               │
│ • 任務狀態顯示（狀態訊息）                                   │
│ • 比對結果卡片列表                                           │
│ • 視覺化圖片（疊圖、熱力圖）                                 │
│ • 圖片預覽對話框                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 關鍵設計特點

### 1. 即時結果更新（Streaming Results）

**設計理念**: 不需要等待所有印鑑比對完成，每個印鑑比對完成後立即更新並顯示。

**實現方式**:
- 使用回調函數 (`task_update_callback`) 在每個印鑑比對完成時立即更新任務記錄
- 前端輪詢時過濾出已完成疊圖的結果（`overlay1_path` 或 `overlay2_path` 存在）
- 即時顯示已完成比對的印鑑結果

**優勢**:
- 用戶體驗更好（不需要等待所有比對完成）
- 可以即時發現問題（某個印鑑比對失敗）

### 2. 線程安全更新

**設計理念**: 多個線程同時更新同一個任務記錄時，確保數據一致性。

**實現方式**:
- 使用 `with_for_update()` 鎖定任務記錄
- 使用字典追蹤結果（以 `seal_index` 為鍵），避免重複
- 重試機制（最多 3 次）處理數據庫衝突

**關鍵代碼**:
```python
task = db_task.query(MultiSealComparisonTask).filter(
    MultiSealComparisonTask.task_uid == task_uid_str
).with_for_update().first()  # 鎖定記錄
```

### 3. 結果完整性保證

**設計理念**: 確保所有印鑑都有比對結果，即使比對失敗也要記錄錯誤。

**實現方式**:
- 比對完成後驗證結果數量
- 識別缺失的印鑑索引
- 重新比對缺失的印鑑（單線程執行）
- 如果重新比對失敗，創建錯誤結果

### 4. 視覺化結果生成

**設計理念**: 不僅提供相似度分數，還提供視覺化的比對結果。

**生成的視覺化**:
- **輸入圖像1**: 去背景後的圖像1（用於比對的實際圖像）
- **輸入圖像2**: 對齊後的印鑑圖像（用於比對的實際圖像）
- **疊圖1**: 圖像1疊在印鑑上（用於視覺檢查對齊效果）
- **疊圖2**: 印鑑疊在圖像1上（用於視覺檢查對齊效果）
- **差異熱力圖**: 顯示兩個圖像的差異區域（紅色表示差異大）

---

## 數據流圖

```
前端狀態管理
├── comparisonResults (比對結果列表)
│   └── 每個元素包含：
│       ├── seal_index
│       ├── similarity
│       ├── is_match
│       ├── overlay1_path
│       ├── overlay2_path
│       ├── heatmap_path
│       ├── input_image1_path
│       ├── input_image2_path
│       └── error
│
├── taskStatus (任務狀態)
│   ├── status: pending | processing | completed | failed
│   ├── total_count
│   └── success_count
│
└── currentTaskUid (當前任務 UID)
    └── 用於輪詢任務結果

↓ 輪詢 API

後端任務記錄 (MultiSealComparisonTask)
├── task_uid
├── status
├── results (JSON 陣列)
│   └── 每個元素對應一個印鑑的比對結果
├── total_count
└── success_count

↓ 結果呈現

前端組件 (MultiSealComparisonResults)
├── 接收 results 陣列
├── 構建圖片 URL
├── 渲染結果卡片
└── 處理圖片點擊預覽
```

---

## 總結

多印鑑比對系統的結果呈現處理具有以下特點：

1. **即時性**: 每個印鑑比對完成後立即更新並顯示，不需要等待所有比對完成
2. **完整性**: 確保所有印鑑都有結果，即使比對失敗也記錄錯誤
3. **視覺化**: 提供多種視覺化結果（疊圖、熱力圖）幫助用戶理解比對結果
4. **線程安全**: 使用數據庫鎖定機制確保多線程更新的一致性
5. **用戶體驗**: 即時反饋、錯誤處理完善

整個流程從輸入準備到結果呈現，每個階段都有明確的 Input、Process 和 Output，形成了完整的數據流和處理鏈。

