# 多印鑑 / PDF 全頁比對流程（以 /multi-seal-test 為主）

本文件只描述 **目前程式碼實際存在** 的主流程：前端 `/multi-seal-test` 搭配後端 `images` API。

## 端到端資料流

```mermaid
flowchart TD
  UI[MultiSealTest_UI] -->|upload| ImagesUpload[POST_/api/v1/images/upload]
  UI -->|detect_seal| DetectSeal[POST_/api/v1/images/{id}/detect-seal]
  UI -->|save_seal_location| SaveSeal[PUT_/api/v1/images/{id}/seal-location]
  UI -->|detect_multiple_seals| DetectMulti[POST_/api/v1/images/{id}/detect-multiple-seals]
  UI -->|save_multiple_seals| SaveMulti[POST_/api/v1/images/{id}/save-multiple-seals]
  UI -->|crop_seals| Crop[POST_/api/v1/images/{id}/crop-seals]
  UI -->|compare_with_seals_task| CompareTask[POST_/api/v1/images/{image1_id}/compare-with-seals]
  UI -->|poll_task_status| TaskStatus[GET_/api/v1/images/tasks/{task_uid}/status]
  UI -->|poll_task_result| TaskResult[GET_/api/v1/images/tasks/{task_uid}]
  UI -->|compare_pdf_task| ComparePdf[POST_/api/v1/images/{image1_id}/compare-pdf]
  UI -->|poll_pdf_status| PdfStatus[GET_/api/v1/images/pdf-tasks/{task_uid}/status]
  UI -->|poll_pdf_result| PdfResult[GET_/api/v1/images/pdf-tasks/{task_uid}]
```

## 重要預設值（程式碼現況）

- **threshold**：0.83
- **maxSeals**：3（UI 可調；PDF 全頁比對另有 max_seals 預設）
- **PDF 全頁比對**：會把圖像1（模板頁）與圖像2（PDF）所有頁面進行自動偵測與比對

## 去背景（背景移除）與印泥顏色

去背景核心在 `backend/core/seal_compare.py` 的 `SealComparator._auto_detect_bounds_and_remove_background()`。\n\n目前印泥顏色僅支援 **紅 / 藍**，並採 **自動判別**（以相對背景色偏 + HSV 區間做前景 mask），避免淡紅/淡藍筆劃被誤刪。

## 結果呈現（前端）

`frontend/src/components/MultiSealComparisonResults.jsx` 會從後端回傳的結果中，顯示每顆印鑑的分數與視覺化圖片（透過 `GET /api/v1/images/multi-seal-comparisons/{filename}` 取圖）。


