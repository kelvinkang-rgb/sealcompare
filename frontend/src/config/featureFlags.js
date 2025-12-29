/**
 * 功能開關配置系統
 * 
 * 透過環境變數控制前端組件的顯示/隱藏
 * 環境變數格式：VITE_FEATURE_<COMPONENT_NAME>
 * 值為 'true' 或 'false'（字符串格式）
 * 
 * 預設行為：所有功能預設啟用（true），未設置時保持可見
 */

// 功能開關常數定義（對應環境變數名稱）
export const FEATURE_FLAGS = {
  // 組件級功能開關
  BATCH_SEAL_ADJUSTMENT: 'BATCH_SEAL_ADJUSTMENT',
  METRICS_EXPLANATION_DIALOG: 'METRICS_EXPLANATION_DIALOG',
  SIMILARITY_HISTOGRAM: 'SIMILARITY_HISTOGRAM',
  COMPARISON_EDIT_DIALOG: 'COMPARISON_EDIT_DIALOG',
  DELETE_CONFIRM_DIALOG: 'DELETE_CONFIRM_DIALOG',
  PROCESSING_STAGES: 'PROCESSING_STAGES',
  VERIFICATION_VIEW: 'VERIFICATION_VIEW',
  IMAGE_MODAL: 'IMAGE_MODAL',
  IMAGE_PREVIEW_DIALOG: 'IMAGE_PREVIEW_DIALOG',
  
  // UI 元素開關
  MASK_STATISTICS: 'MASK_STATISTICS',
  TIMING_DETAILS: 'TIMING_DETAILS',
  TASK_TIMING_STATISTICS: 'TASK_TIMING_STATISTICS',
  ADVANCED_SETTINGS: 'ADVANCED_SETTINGS',
  MAX_SEALS_SETTING: 'MAX_SEALS_SETTING',
  THRESHOLD_SETTING: 'THRESHOLD_SETTING',
  MASK_WEIGHTS_SETTING: 'MASK_WEIGHTS_SETTING',
  ALIGNMENT_TIMING_DETAILS: 'ALIGNMENT_TIMING_DETAILS',
}

/**
 * 從環境變數讀取功能開關值
 * @param {string} featureName - 功能名稱（FEATURE_FLAGS 中的鍵）
 * @returns {boolean} 功能是否啟用
 */
function getFeatureFlag(featureName) {
  const envKey = `VITE_FEATURE_${featureName}`
  const envValue = import.meta.env[envKey]
  
  // 如果環境變數未設置，預設返回 true（啟用）
  if (envValue === undefined) {
    return true
  }
  
  // 轉換為布林值
  return envValue === 'true' || envValue === true
}

/**
 * 統一配置對象
 * 包含所有功能開關的當前狀態
 */
export const featureConfig = {
  batchSealAdjustment: getFeatureFlag(FEATURE_FLAGS.BATCH_SEAL_ADJUSTMENT),
  metricsExplanationDialog: getFeatureFlag(FEATURE_FLAGS.METRICS_EXPLANATION_DIALOG),
  similarityHistogram: getFeatureFlag(FEATURE_FLAGS.SIMILARITY_HISTOGRAM),
  comparisonEditDialog: getFeatureFlag(FEATURE_FLAGS.COMPARISON_EDIT_DIALOG),
  deleteConfirmDialog: getFeatureFlag(FEATURE_FLAGS.DELETE_CONFIRM_DIALOG),
  processingStages: getFeatureFlag(FEATURE_FLAGS.PROCESSING_STAGES),
  verificationView: getFeatureFlag(FEATURE_FLAGS.VERIFICATION_VIEW),
  imageModal: getFeatureFlag(FEATURE_FLAGS.IMAGE_MODAL),
  imagePreviewDialog: getFeatureFlag(FEATURE_FLAGS.IMAGE_PREVIEW_DIALOG),
  
  // UI 元素開關
  maskStatistics: getFeatureFlag(FEATURE_FLAGS.MASK_STATISTICS),
  timingDetails: getFeatureFlag(FEATURE_FLAGS.TIMING_DETAILS),
  taskTimingStatistics: getFeatureFlag(FEATURE_FLAGS.TASK_TIMING_STATISTICS),
  advancedSettings: getFeatureFlag(FEATURE_FLAGS.ADVANCED_SETTINGS),
  maxSealsSetting: getFeatureFlag(FEATURE_FLAGS.MAX_SEALS_SETTING),
  thresholdSetting: getFeatureFlag(FEATURE_FLAGS.THRESHOLD_SETTING),
  maskWeightsSetting: getFeatureFlag(FEATURE_FLAGS.MASK_WEIGHTS_SETTING),
  alignmentTimingDetails: getFeatureFlag(FEATURE_FLAGS.ALIGNMENT_TIMING_DETAILS),
}

/**
 * 檢查功能是否啟用（供非 React 組件使用）
 * @param {string} featureName - 功能名稱（FEATURE_FLAGS 中的鍵）
 * @returns {boolean} 功能是否啟用
 */
export function isFeatureEnabled(featureName) {
  const envKey = `VITE_FEATURE_${featureName}`
  const envValue = import.meta.env[envKey]
  
  // 如果環境變數未設置，預設返回 true（啟用）
  if (envValue === undefined) {
    return true
  }
  
  // 轉換為布林值
  return envValue === 'true' || envValue === true
}

/**
 * React Hook：獲取功能開關狀態（供 React 組件使用）
 * @param {string} featureName - 功能名稱（FEATURE_FLAGS 中的鍵）
 * @returns {boolean} 功能是否啟用
 * 
 * @example
 * ```jsx
 * function MyComponent() {
 *   const showMetrics = useFeatureFlag('METRICS_EXPLANATION_DIALOG')
 *   
 *   return (
 *     <div>
 *       {showMetrics && <MetricsExplanationDialog />}
 *     </div>
 *   )
 * }
 * ```
 */
export function useFeatureFlag(featureName) {
  // 在構建時注入，運行時值不會變化，所以可以直接返回
  // 如果需要運行時動態變化，可以使用 useState，但這裡不需要
  return isFeatureEnabled(featureName)
}

export default featureConfig

