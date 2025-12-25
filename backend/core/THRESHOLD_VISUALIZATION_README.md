# Threshold 視覺化說明

## gray_diff 與 threshold 的單位和範圍

### gray_diff (灰度差異)
- **單位**: 像素灰度值的絕對差值
- **範圍**: 0-255 (uint8 格式)
- **計算方式**: `gray_diff = cv2.absdiff(gray1, gray2)`
  - `gray_diff[i,j] = |gray1[i,j] - gray2[i,j]|`
- **意義**:
  - `0`: 兩個圖像在該像素位置完全相同
  - `255`: 兩個圖像在該像素位置完全相反（例如：黑 vs 白）
  - 值越大，表示差異越大

### diff_threshold (差異閾值)
- **單位**: 像素灰度值的絕對差值
- **範圍**: 0-255
- **預設值**: 100 (約為灰度範圍的 39%)
- **判定規則**: `pixel_diff_mask = (gray_diff > diff_threshold) & overlap_mask`
- **影響**:
  - **threshold 越小** (例如 30-50): 越敏感，更多像素被標記為差異
  - **threshold 中等** (例如 100): 平衡敏感度和嚴格度，適合一般情況
  - **threshold 越大** (例如 150-200): 越嚴格，只有明顯差異的像素被標記

## 運行視覺化腳本

### 前置需求
```bash
pip install matplotlib numpy opencv-python
```

### 執行方式
```bash
cd backend/core
python visualize_threshold.py
```

### 輸出結果
腳本會在 `backend/core/threshold_visualization/` 目錄下生成以下文件：

1. **threshold_explanation.png**: 說明圖，展示 threshold 的概念和影響
2. **threshold_comparison.png**: 綜合對比圖，展示多個 threshold 值的效果
3. **threshold_{value}_comparison.png**: 每個 threshold 值的單獨對比圖
   - `threshold_30_comparison.png`
   - `threshold_50_comparison.png`
   - `threshold_100_comparison.png`
   - `threshold_150_comparison.png`
   - `threshold_200_comparison.png`
4. **test_image1.png** 和 **test_image2.png**: 用於測試的示例圖像

### 視覺化內容
每個對比圖包含：
- **原始圖像對比**: 並排顯示兩個測試圖像
- **灰度差異圖**: 使用熱力圖顯示差異值 (0-255)
- **Pixel Diff Mask**: 標記被判定為差異的像素 (白色區域)
- **原圖 + 差異標記**: 在原圖上用紅色標記差異區域
- **差異值分佈直方圖**: 顯示差異值的統計分佈，並標記 threshold 位置

## Threshold 選擇建議

### 根據應用場景選擇

| Threshold 值 | 適用場景 | 特點 |
|-------------|---------|------|
| 30-50 | 高精度比對，需要檢測細微差異 | 非常敏感，可能產生較多誤報 |
| 80-120 | 一般比對場景（推薦） | 平衡敏感度和準確度 |
| 150-200 | 寬鬆比對，只關注明顯差異 | 嚴格，可能忽略細微差異 |

### 根據圖像品質調整

- **高品質掃描圖像**: 可以使用較小的 threshold (50-80)
- **一般品質照片**: 使用中等 threshold (100-120)
- **低品質或噪點較多的圖像**: 使用較大的 threshold (120-150)

## 範例說明

假設有兩個像素：
- 圖像1 的像素值: 150
- 圖像2 的像素值: 180
- gray_diff = |150 - 180| = 30

不同 threshold 的判定結果：
- threshold = 20: ✅ 判定為差異 (30 > 20)
- threshold = 30: ❌ 不判定為差異 (30 不大於 30)
- threshold = 50: ❌ 不判定為差異 (30 < 50)

## 相關程式碼位置

- **計算位置**: `backend/core/overlay.py` 第 190-193 行
- **視覺化腳本**: `backend/core/visualize_threshold.py`
- **Mask 統計計算**: `backend/core/overlay.py` 的 `calculate_mask_statistics()` 函數
- **Mask 相似度計算**: `backend/core/overlay.py` 的 `calculate_mask_based_similarity()` 函數

