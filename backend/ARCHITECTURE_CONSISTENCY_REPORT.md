# 後端架構一致性檢查報告

**檢查日期**: 2024年
**檢查範圍**: `backend/app/` 目錄下的所有 Python 文件

## 執行摘要

本次檢查發現了 **5 個主要問題**，涉及架構分層、異常處理、導入規範等方面。大部分問題屬於架構設計不一致，需要重構以符合分層架構原則。

---

## 1. 代碼結構檢查 ✅

### 1.1 文件組織
**狀態**: ✅ **符合規範**

文件組織符合架構設計：
- `app/api/` - API 路由層（4個文件）
- `app/services/` - 業務邏輯層（2個文件）
- `app/utils/` - 工具類（2個文件）
- `app/models.py` - 數據模型
- `app/schemas.py` - Pydantic 驗證模型
- `app/config.py` - 配置管理
- `app/database.py` - 數據庫連接
- `app/main.py` - 應用入口

### 1.2 模塊劃分
**狀態**: ✅ **職責清晰**

各模塊職責劃分清晰：
- API 層負責路由和請求驗證
- Service 層負責業務邏輯
- Utils 層提供工具函數
- Models 定義數據結構

---

## 2. 架構分層檢查 ❌

### 2.1 API 層直接操作數據庫（違反分層架構）

**嚴重程度**: 🔴 **高**

**問題描述**: API 層不應直接操作數據庫，所有數據庫操作應通過 Service 層進行。

#### 問題位置：

1. **`backend/app/api/statistics.py`** (第 23-43 行)
   ```python
   base_query = db.query(Comparison).filter(Comparison.deleted_at.is_(None))
   total_comparisons = base_query.count()
   match_count = base_query.filter(Comparison.is_match == True).count()
   # ... 直接使用 db.query()
   ```
   **影響**: 統計邏輯應該在 `ComparisonService` 中

2. **`backend/app/api/visualizations.py`** (第 27-37, 63-70, 101-110 行)
   ```python
   comparison = db.query(Comparison).filter(...).first()
   vis = db.query(ComparisonVisualization).filter(...).first()
   ```
   **影響**: 視覺化查詢邏輯應該在 `ComparisonService` 中

3. **`backend/app/api/images.py`** (第 321-322, 704, 735 行)
   ```python
   db.add(task)
   db.commit()
   task = db.query(MultiSealComparisonTask).filter(...).first()
   ```
   **影響**: `MultiSealComparisonTask` 的管理應該在 `ImageService` 中

4. **`backend/app/api/comparisons.py`** (第 53, 254, 271 行)
   ```python
   db_comparison = db.query(Comparison).filter(Comparison.id == comp_id).first()
   db.commit()
   ```
   **影響**: 後台任務中的數據庫操作應該通過 Service 層

**修復建議**:
- 在 `ComparisonService` 中添加 `get_statistics()` 方法
- 在 `ComparisonService` 中添加 `get_comparison_visualization(comparison_id, vis_type)` 方法
- 在 `ImageService` 中添加管理 `MultiSealComparisonTask` 的方法
- 修改所有 API 端點，移除直接數據庫操作

---

## 3. 命名規範檢查 ✅

### 3.1 類名
**狀態**: ✅ **符合規範**

所有類名使用 PascalCase：
- `ImageService` ✅
- `ComparisonService` ✅
- `Image` ✅
- `Comparison` ✅
- `ComparisonStatus` ✅
- `VisualizationType` ✅

### 3.2 函數/方法名
**狀態**: ✅ **符合規範**

所有函數和方法名使用 snake_case：
- `create_image()` ✅
- `get_image()` ✅
- `detect_seal()` ✅
- `process_comparison()` ✅
- `_detect_by_contours_fast()` ✅ (私有方法使用下劃線前綴)

### 3.3 變量名
**狀態**: ✅ **符合規範**

所有變量名使用 snake_case：
- `db_image` ✅
- `comparison_id` ✅
- `upload_file` ✅

### 3.4 文件名
**狀態**: ✅ **符合規範**

所有文件名使用 snake_case：
- `image_service.py` ✅
- `comparison_service.py` ✅
- `seal_detector.py` ✅

---

## 4. 導入語句檢查 ⚠️

### 4.1 重複導入

**嚴重程度**: 🟡 **中**

**位置**: `backend/app/services/image_service.py`

**問題**:
```python
# 第 7 行
from typing import Optional, Dict, List, Tuple, Callable

# 第 29 行（重複）
from typing import Dict, Optional, List, Tuple
```

**修復建議**: 移除第 29 行的重複導入，保留第 7 行的完整導入。

### 4.2 使用 sys.path.insert 導入 core 模塊

**嚴重程度**: 🟡 **中**

**問題描述**: 使用動態修改 `sys.path` 來導入 `core` 模塊，這不是最佳實踐。

**位置**:
- `backend/app/services/image_service.py` (第 20-23 行)
- `backend/app/services/comparison_service.py` (第 15-19 行)
- `backend/app/utils/seal_detector.py` (多處)

**當前實現**:
```python
import sys
from pathlib import Path as PathLib
core_path = PathLib(__file__).parent.parent.parent / "core"
sys.path.insert(0, str(core_path))
from seal_compare import SealComparator
```

**修復建議**:
1. **方案 1（推薦）**: 將 `core/` 目錄添加到 `PYTHONPATH` 環境變量
2. **方案 2**: 在 `backend/` 目錄下創建 `__init__.py`，使 `core` 成為包的一部分，使用相對導入
3. **方案 3**: 使用 `importlib` 動態導入（但不如方案 1 和 2 清晰）

---

## 5. 異常處理檢查 ❌

### 5.1 Service 層使用 HTTPException

**嚴重程度**: 🔴 **高**

**問題描述**: Service 層不應使用 `HTTPException`，這是 API 層的職責。Service 層應該拋出自定義業務異常，由 API 層轉換為 HTTP 響應。

**位置**: `backend/app/services/image_service.py`

**問題統計**: 發現 **30 處**使用 `HTTPException`

**示例**:
```python
# 第 141 行
raise HTTPException(status_code=404, detail="圖像不存在")

# 第 187 行
raise HTTPException(status_code=400, detail="邊界框格式錯誤")
```

**修復建議**:
1. 創建 `backend/app/exceptions.py`，定義業務異常類：
   ```python
   class ImageNotFoundError(Exception):
       """圖像不存在異常"""
       pass
   
   class InvalidBboxError(Exception):
       """無效的邊界框異常"""
       pass
   
   class ImageFileNotFoundError(Exception):
       """圖像文件不存在異常"""
       pass
   ```

2. 修改 `ImageService`，使用業務異常替代 `HTTPException`

3. 在 `main.py` 或 API 層添加異常處理器，將業務異常轉換為 HTTP 響應：
   ```python
   @app.exception_handler(ImageNotFoundError)
   async def image_not_found_handler(request: Request, exc: ImageNotFoundError):
       return JSONResponse(
           status_code=404,
           content={"detail": str(exc)}
       )
   ```

---

## 6. 其他發現

### 6.1 後台任務中的數據庫操作

**位置**: `backend/app/api/comparisons.py` (第 40-65 行, 258-283 行)

**問題**: 後台任務函數內部直接操作數據庫，雖然創建了新的會話，但應該通過 Service 層進行操作。

**建議**: 將後台任務中的數據庫操作封裝到 Service 方法中。

### 6.2 缺少 Service 層方法

以下功能缺少對應的 Service 方法：

1. **統計功能**: `ComparisonService.get_statistics()` - 用於 `statistics.py`
2. **視覺化查詢**: `ComparisonService.get_comparison_visualization()` - 用於 `visualizations.py`
3. **任務管理**: `ImageService` 中應有管理 `MultiSealComparisonTask` 的方法

---

## 修復優先級

### 高優先級（必須修復）
1. ✅ **API 層直接操作數據庫** - 違反分層架構原則
2. ✅ **Service 層使用 HTTPException** - 違反分層架構原則

### 中優先級（建議修復）
3. ⚠️ **重複導入語句** - 代碼整潔性問題
4. ⚠️ **使用 sys.path.insert** - 導入規範問題

### 低優先級（可選優化）
5. ℹ️ **後台任務中的數據庫操作** - 可以通過重構改進

---

## 修復計劃

### 階段 1: 創建業務異常類
1. 創建 `backend/app/exceptions.py`
2. 定義所有業務異常類

### 階段 2: 修復 Service 層
1. 修改 `ImageService`，移除 `HTTPException`，使用業務異常
2. 在 `ComparisonService` 中添加 `get_statistics()` 方法
3. 在 `ComparisonService` 中添加 `get_comparison_visualization()` 方法
4. 在 `ImageService` 中添加 `MultiSealComparisonTask` 管理方法

### 階段 3: 修復 API 層
1. 修改 `statistics.py`，使用 `ComparisonService.get_statistics()`
2. 修改 `visualizations.py`，使用 `ComparisonService.get_comparison_visualization()`
3. 修改 `images.py`，使用 `ImageService` 的任務管理方法
4. 修改 `comparisons.py`，後台任務通過 Service 層操作

### 階段 4: 清理導入
1. 移除重複導入
2. 改進 `core` 模塊導入方式

### 階段 5: 添加異常處理器
1. 在 `main.py` 中添加異常處理器
2. 將業務異常轉換為 HTTP 響應

---

## 檢查統計

- **總文件數**: 10
- **檢查項目**: 5 大類
- **發現問題**: 5 個
- **高優先級問題**: 2 個
- **中優先級問題**: 2 個
- **低優先級問題**: 1 個

---

## 結論

後端代碼整體結構良好，命名規範一致，但在架構分層方面存在違反設計原則的問題。主要問題是 API 層直接操作數據庫和 Service 層使用 HTTPException，這些都需要重構以符合分層架構原則。

建議按照修復計劃逐步修復，優先處理高優先級問題，確保架構分層清晰，職責明確。

