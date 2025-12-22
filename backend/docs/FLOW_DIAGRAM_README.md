# 印鑑比對系統流程圖說明

## 概述

本文檔說明印鑑比對系統的完整流程，包括每個階段的輸入（Input）、處理（Process）和輸出（Output），以及潛在的問題和優化建議。

## 流程階段詳解

### 階段1: 初始化
- **Input**: `comparison_id`, `enable_rotation_search`, `enable_translation_search`
- **Process**: 創建 `processing_stages` 結構，設置狀態為 `PROCESSING`
- **Output**: `processing_stages` 結構，狀態更新到資料庫

### 階段2: 載入圖像
- **Input**: `db_comparison.image1.file_path`, `image2.file_path`
- **Process**: 
  - 標準化路徑，驗證文件存在
  - 讀取裁切區域信息（bbox1, bbox2，可選）
  - 使用 `cv2.imread()` 載入圖像
  - 根據 bbox 裁切圖像（如有）
- **Output**: `img1_cropped`, `img2_cropped`

### 階段3: 圖像1去背景（已移除對齊步驟）
- **Input**: `img1_cropped`
- **Process**: `_auto_detect_bounds_and_remove_background()`
  - 自動檢測邊界框
  - 移除背景（設為白色）
- **Output**: `img1_no_bg` (去背景後的圖像1)
- **保存**: `image1_cropped_path` → `image1_for_comparison`

**注意：不再調用 `_align_image1()`**（函數定義保留但已移除調用）

### 階段4: 圖像2去背景和對齊（相對於圖像1優化）
- **Input**: 
  - `img2_cropped` (待處理圖像)
  - `img1_no_bg` (參考圖像，已去背景)
- **Process**: 
  1. **去背景**: `_auto_detect_bounds_and_remove_background()` → `img2_no_bg`
  2. **對齊優化**: `_align_image2_to_image1(img1_no_bg, img2_no_bg)`
     - **階段1: 粗搜索**
       - 縮放比例: 0.2（優化後，加快速度）
       - 角度步長: 3度（優化後，提高精度）
       - 偏移步長: 8像素（優化後，提高精度）
       - 提前終止: 如果找到相似度 > 0.95 的候選，提前結束
     - **階段2: 完整尺寸評估**
       - 動態調整候選數量: 根據分數差異決定（3-7個）
       - 使用 `_fast_rotation_match()` 進行評估
     - **階段3: 細搜索優化**
       - 角度範圍: ±2度，步長 0.5度
       - 偏移範圍: ±10像素，步長 1像素
- **Output**: 
  - `img2_aligned` (對齊後的圖像2)
  - `rotation_angle` (旋轉角度)
  - `translation_offset` (平移偏移)
  - `alignment_similarity` (對齊相似度)
- **保存**: 
  - `alignment_optimization` 到 `details`
  - `image2_cropped_path` → `image2_for_comparison`

### 階段5: 計算相似度（圖像已對齊，不需要旋轉/平移搜索）
- **Input**: `image1_for_comparison`, `image2_for_comparison`
- **Process**: `compare_files()` → `compare_images()`
  1. 載入圖像並轉換為灰度圖
  2. **直接使用原始尺寸**（移除尺寸調整以避免圖像變形）
  3. 計算多種相似度指標：
     - SSIM (結構相似性指數)
     - Template Match (模板匹配)
     - Pixel Similarity (像素相似度)
     - Histogram Similarity (直方圖相似度)
  4. 加權組合計算最終相似度：
     - SSIM: 0.4
     - Template: 0.3
     - Pixel: 0.2
     - Histogram: 0.1
- **Output**: 
  - `similarity` (最終相似度)
  - `is_match` (是否匹配)
  - `details` (包含所有指標)

**優化說明**:
- ✓ 移除尺寸調整（避免圖像變形）
- ✓ 移除未使用的指標計算（edge_similarity, exact_match_ratio, mse_similarity）
- ✓ 圖像已對齊，不需要旋轉/平移搜索

### 階段6: 保存校正後圖像
- **Input**: `img1_corrected`, `img2_corrected` (從 `compare_files` 返回)
- **Process**: 
  - `cv2.imwrite()` 保存到 `corrected_images` 目錄
  - 更新 `db_comparison.details`，從 `alignment_optimization` 提取值到頂層
- **Output**: `image1_corrected_path`, `image2_corrected_path`

### 階段7: 生成視覺化圖表
- **Input**: 
  - `image1_path_for_viz` (優先使用校正後的圖像1)
  - `image2_path_for_viz` (裁切後的圖像2)
  - `image2_corrected_path` (校正後的圖像2)
- **Process**: `_generate_visualizations()`
  - `create_correction_comparison()` - 並排對比圖
  - `create_difference_heatmap()` - 差異熱圖
  - `create_overlay_image()` - 疊加圖像（overlay1, overlay2）
- **Output**: `comparison_image`, `heatmap`, `overlay1`, `overlay2`

### 階段8: 完成
- **Input**: 所有處理結果
- **Process**: 更新狀態為 `COMPLETED`，保存所有結果到資料庫
- **Output**: 完整的比對記錄（包含所有 details）

## 潛在問題和優化建議

### 問題1: 圖像1未對齊
**問題描述**:
- 圖像1只進行了去背景處理，沒有對齊（`_align_image1` 已移除調用）
- 這可能導致圖像1和圖像2的基準不一致，影響比對準確度

**影響**:
- 如果圖像1本身有旋轉或偏移，圖像2對齊到圖像1後，兩者的基準可能不一致
- 可能導致相似度計算不準確

**建議**:
- 考慮對圖像1也進行基本對齊（中心對齊），或確保圖像1已經是標準化狀態
- 或者保留 `_align_image1` 的調用，但簡化其邏輯（只做中心對齊，不做旋轉）

### 問題2: 對齊搜索可能較慢
**問題描述**:
- `_align_image2_to_image1` 使用三階段搜索，雖然已優化（縮放0.2，角度3度，偏移8px）
- 但對於大圖像，粗搜索階段仍可能較慢

**影響**:
- 處理時間可能較長，特別是對於高分辨率圖像

**建議**:
- 考慮使用多線程或GPU加速
- 或進一步優化搜索策略（如使用更激進的縮放比例，或減少搜索範圍）
- 考慮使用特徵點匹配（如 SIFT/ORB）來快速定位大致位置

### 問題3: 相似度計算未考慮尺寸差異
**問題描述**:
- 當前流程移除了尺寸調整，直接使用原始尺寸計算相似度
- 如果兩個圖像尺寸差異很大，可能影響相似度計算準確度

**影響**:
- SSIM 和 Template Match 等方法對尺寸差異敏感
- 如果圖像1和圖像2尺寸差異很大，相似度可能不準確

**建議**:
- 考慮在計算相似度前進行智能尺寸標準化（保持長寬比）
- 或者使用對尺寸不敏感的相似度指標（如直方圖相似度）
- 或者在 `_fast_rotation_match` 中處理尺寸差異

### 問題4: 圖像保存重複
**問題描述**:
- 圖像在載入階段保存一次（`image1_cropped_path`, `image2_cropped_path`）
- 在保存階段又保存一次（`image1_corrected_path`, `image2_corrected_path`）
- 實際上，`compare_files` 返回的 `img1_corrected` 和 `img2_corrected` 可能與已保存的圖像相同

**影響**:
- 浪費存儲空間
- 可能導致混淆（哪個文件是最終結果？）

**建議**:
- 統一保存邏輯，避免重複保存相同圖像
- 明確區分中間結果和最終結果
- 考慮使用符號鏈接或引用，而不是複製文件

### 問題5: 缺少錯誤恢復機制
**問題描述**:
- 如果某個階段失敗，整個流程會中斷
- 沒有部分結果的保存機制

**建議**:
- 實現階段性的結果保存
- 添加錯誤恢復機制，允許從失敗的階段重新開始
- 考慮使用事務機制，確保數據一致性

### 問題6: 記憶體使用可能較高
**問題描述**:
- 多個圖像副本同時存在於記憶體中（原始、裁切、去背景、對齊後等）
- 對於大圖像，可能導致記憶體不足

**建議**:
- 及時釋放不需要的圖像副本
- 考慮使用流式處理，避免同時載入所有圖像
- 使用圖像壓縮或降採樣來減少記憶體使用

## 優化建議總結

### 短期優化（容易實施）
1. **統一圖像保存邏輯**：避免重複保存相同圖像
2. **及時釋放記憶體**：處理完後立即釋放不需要的圖像副本
3. **添加進度報告**：更詳細的進度信息，幫助用戶了解處理狀態

### 中期優化（需要一些開發工作）
1. **圖像1基本對齊**：實現簡化的 `_align_image1`（只做中心對齊）
2. **智能尺寸標準化**：在保持長寬比的前提下進行尺寸調整
3. **緩存中間結果**：減少重複計算（如去背景結果）

### 長期優化（需要較大改動）
1. **多線程/GPU加速**：加速對齊搜索和相似度計算
2. **異步處理**：提高並發性能
3. **錯誤恢復機制**：實現階段性的結果保存和恢復
4. **特徵點匹配**：使用 SIFT/ORB 等特徵點匹配來快速定位大致位置

## 最新變更記錄

### 2024-12-22 優化後版本
- ✓ **圖像1**：只去背景，不對齊（`_align_image1` 已移除調用）
- ✓ **圖像2**：去背景 + 對齊優化（使用 `_align_image2_to_image1`）
- ✓ **對齊搜索優化**：
  - 縮放比例: 0.3 → 0.2（加快速度）
  - 角度步長: 5度 → 3度（提高精度）
  - 偏移步長: 10像素 → 8像素（提高精度）
  - 提前終止機制（相似度 > 0.95）
  - 動態候選數量（3-7個）
- ✓ **移除尺寸調整**：直接使用原始尺寸計算相似度（避免圖像變形）
- ✓ **移除未使用的指標計算**：edge_similarity, exact_match_ratio, mse_similarity
- ✓ **移除未使用的函數**：約 1000+ 行代碼

## 流程圖文件

詳細的流程圖請參考：`comparison_flow_diagram.svg`

流程圖包含：
- 每個階段的詳細輸入、處理和輸出
- 潛在問題標註
- 優化建議說明
- 最新變更記錄
