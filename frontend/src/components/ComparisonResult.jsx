import React, { useEffect, useState } from 'react'
import {
  Box,
  Paper,
  Typography,
  CircularProgress,
  Alert,
  Button,
} from '@mui/material'
import { Refresh as RefreshIcon } from '@mui/icons-material'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { comparisonAPI } from '../services/api'
import VerificationView from './VerificationView'
import ProcessingStages from './ProcessingStages'

function ComparisonResult({ comparisonId, onResetComparison }) {
  const [pollingInterval, setPollingInterval] = useState(1000)
  const queryClient = useQueryClient()

  const { data: comparison, isLoading, error } = useQuery({
    queryKey: ['comparison', comparisonId],
    queryFn: () => comparisonAPI.get(comparisonId),
    refetchInterval: pollingInterval,
  })

  const { data: status } = useQuery({
    queryKey: ['comparison-status', comparisonId],
    queryFn: () => comparisonAPI.getStatus(comparisonId),
    refetchInterval: pollingInterval,
    enabled: !!comparisonId,
  })

  // 重試比對的 mutation
  const retryMutation = useMutation({
    mutationFn: () => comparisonAPI.retry(comparisonId, true, false),
    onSuccess: () => {
      // 重新獲取比對數據
      queryClient.invalidateQueries({ queryKey: ['comparison', comparisonId] })
      queryClient.invalidateQueries({ queryKey: ['comparison-status', comparisonId] })
      setPollingInterval(1000) // 重新開始輪詢
    },
  })

  useEffect(() => {
    if (status?.status === 'completed' || status?.status === 'failed') {
      setPollingInterval(0) // 停止輪詢
    }
  }, [status])

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    )
  }

  if (error) {
    return <Alert severity="error">載入失敗</Alert>
  }

  if (!comparison) {
    return null
  }

  const isProcessing = status?.status === 'processing' || status?.status === 'pending'

  return (
    <Paper sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">
          比對結果
        </Typography>
        {onResetComparison && (
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={onResetComparison}
            disabled={isProcessing}
          >
            重新比對
          </Button>
        )}
      </Box>

      {isProcessing && (
        <Box sx={{ mb: 2 }}>
          <Alert 
            severity="info" 
            sx={{ mb: 2 }}
            action={
              <Button
                size="small"
                onClick={() => retryMutation.mutate()}
                disabled={retryMutation.isPending}
              >
                {retryMutation.isPending ? '重試中...' : '重試'}
              </Button>
            }
          >
            處理中... {status?.progress != null ? `${status.progress}%` : '--'} {status?.message && `- ${status.message}`}
            {comparison.details?.processing_stages?.current_stage === 'loading_images' && 
             comparison.details?.processing_stages?.stages?.[1]?.progress === 10 && (
              <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                如果長時間卡在載入圖像階段，請點擊重試按鈕
              </Typography>
            )}
          </Alert>
          {comparison.details?.processing_stages && (
            <ProcessingStages processingStages={comparison.details.processing_stages} />
          )}
        </Box>
      )}

      {comparison.status === 'completed' && (
        <>
          <Box sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
              {comparison.rotation_angle !== null && comparison.rotation_angle !== undefined && comparison.rotation_angle !== 0 && (
                <Typography variant="body2">
                  旋轉角度: <strong>{comparison.rotation_angle.toFixed(2)}°</strong>
                </Typography>
              )}
              {comparison.translation_offset && comparison.translation_offset.x !== 0 && comparison.translation_offset.y !== 0 && (
                <Typography variant="body2">
                  平移偏移: <strong>X: {comparison.translation_offset.x}px, Y: {comparison.translation_offset.y}px</strong>
                </Typography>
              )}
              {comparison.improvement !== null && comparison.improvement !== undefined && (
                <Typography variant="body2">
                  改善幅度: <strong>{(comparison.improvement * 100).toFixed(2)}%</strong>
                </Typography>
              )}
              {/* 顯示對齊時間詳情 */}
              {comparison.details?.alignment_optimization?.timing && (
                <Box sx={{ mt: 1, p: 1, bgcolor: 'background.default', borderRadius: 1 }}>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                    圖像2處理時間詳情:
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.25, pl: 1 }}>
                    {/* 去背景詳細步驟 */}
                    {comparison.details.alignment_optimization.timing.step1_convert_to_gray !== undefined && (
                      <Typography variant="caption" color="text.secondary" sx={{ pl: 1, fontStyle: 'italic' }}>
                        去背景-步驟1（轉換灰度）: {(comparison.details.alignment_optimization.timing.step1_convert_to_gray * 1000).toFixed(2)}ms
                      </Typography>
                    )}
                    {comparison.details.alignment_optimization.timing.step2_detect_bg_color !== undefined && (
                      <Typography variant="caption" color="text.secondary" sx={{ pl: 1, fontStyle: 'italic' }}>
                        去背景-步驟2（偵測背景色）: {(comparison.details.alignment_optimization.timing.step2_detect_bg_color * 1000).toFixed(2)}ms
                      </Typography>
                    )}
                    {comparison.details.alignment_optimization.timing.step3_otsu_threshold !== undefined && (
                      <Typography variant="caption" color="text.secondary" sx={{ pl: 1, fontStyle: 'italic' }}>
                        去背景-步驟3（OTSU二值化）: {(comparison.details.alignment_optimization.timing.step3_otsu_threshold * 1000).toFixed(2)}ms
                      </Typography>
                    )}
                    {comparison.details.alignment_optimization.timing.step4_combine_masks !== undefined && (
                      <Typography variant="caption" color="text.secondary" sx={{ pl: 1, fontStyle: 'italic' }}>
                        去背景-步驟4（結合遮罩）: {(comparison.details.alignment_optimization.timing.step4_combine_masks * 1000).toFixed(2)}ms
                      </Typography>
                    )}
                    {comparison.details.alignment_optimization.timing.step5_morphology_bg !== undefined && (
                      <Typography variant="caption" color="text.secondary" sx={{ pl: 1, fontStyle: 'italic' }}>
                        去背景-步驟5（形態學處理背景）: {(comparison.details.alignment_optimization.timing.step5_morphology_bg * 1000).toFixed(2)}ms
                      </Typography>
                    )}
                    {comparison.details.alignment_optimization.timing.step6_contour_detection !== undefined && (
                      <Typography variant="caption" color="text.secondary" sx={{ pl: 1, fontStyle: 'italic' }}>
                        去背景-步驟6（輪廓偵測）: {(comparison.details.alignment_optimization.timing.step6_contour_detection * 1000).toFixed(2)}ms
                      </Typography>
                    )}
                    {comparison.details.alignment_optimization.timing.step7_calculate_bbox !== undefined && (
                      <Typography variant="caption" color="text.secondary" sx={{ pl: 1, fontStyle: 'italic' }}>
                        去背景-步驟7（計算邊界框）: {(comparison.details.alignment_optimization.timing.step7_calculate_bbox * 1000).toFixed(2)}ms
                      </Typography>
                    )}
                    {comparison.details.alignment_optimization.timing.step8_crop_image !== undefined && (
                      <Typography variant="caption" color="text.secondary" sx={{ pl: 1, fontStyle: 'italic' }}>
                        去背景-步驟8（裁切圖像）: {(comparison.details.alignment_optimization.timing.step8_crop_image * 1000).toFixed(2)}ms
                      </Typography>
                    )}
                    {comparison.details.alignment_optimization.timing.step9_remove_bg_final !== undefined && (
                      <Typography variant="caption" color="text.secondary" sx={{ pl: 1, fontStyle: 'italic' }}>
                        去背景-步驟9（最終移除背景）: {(comparison.details.alignment_optimization.timing.step9_remove_bg_final * 1000).toFixed(2)}ms
                      </Typography>
                    )}
                    {comparison.details.alignment_optimization.timing.remove_background_total !== undefined && (
                      <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 'bold', mt: 0.5 }}>
                        去背景總時間: {(comparison.details.alignment_optimization.timing.remove_background_total * 1000).toFixed(2)}ms
                      </Typography>
                    )}
                    {/* 對齊詳細步驟 */}
                    {comparison.details.alignment_optimization.timing.stage1_translation_coarse !== undefined && (
                      <Typography variant="caption" color="text.secondary">
                        對齊-階段1（平移粗調）: {(comparison.details.alignment_optimization.timing.stage1_translation_coarse * 1000).toFixed(2)}ms
                      </Typography>
                    )}
                    {comparison.details.alignment_optimization.timing.stage2_rotation_coarse !== undefined && (
                      <Typography variant="caption" color="text.secondary">
                        對齊-階段2（旋轉粗調）: {(comparison.details.alignment_optimization.timing.stage2_rotation_coarse * 1000).toFixed(2)}ms
                      </Typography>
                    )}
                    {comparison.details.alignment_optimization.timing.stage3_translation_fine !== undefined && (
                      <Typography variant="caption" color="text.secondary">
                        對齊-階段3（平移細調）: {(comparison.details.alignment_optimization.timing.stage3_translation_fine * 1000).toFixed(2)}ms
                      </Typography>
                    )}
                    {comparison.details.alignment_optimization.timing.stage4_rotation_fine !== undefined && (
                      <Typography variant="caption" color="text.secondary">
                        對齊-階段4a（旋轉細調）: {(comparison.details.alignment_optimization.timing.stage4_rotation_fine * 1000).toFixed(2)}ms
                      </Typography>
                    )}
                    {comparison.details.alignment_optimization.timing.stage4_translation_fine !== undefined && (
                      <Typography variant="caption" color="text.secondary">
                        對齊-階段4b（平移細調）: {(comparison.details.alignment_optimization.timing.stage4_translation_fine * 1000).toFixed(2)}ms
                      </Typography>
                    )}
                    {comparison.details.alignment_optimization.timing.stage4_total !== undefined && comparison.details.alignment_optimization.timing.stage4_total > 0 && (
                      <Typography variant="caption" color="text.secondary">
                        對齊-階段4（總計）: {(comparison.details.alignment_optimization.timing.stage4_total * 1000).toFixed(2)}ms
                      </Typography>
                    )}
                    {/* 向後兼容：顯示舊的 remove_background 字段 */}
                    {comparison.details.alignment_optimization.timing.remove_background !== undefined && 
                     !comparison.details.alignment_optimization.timing.remove_background_total && (
                      <Typography variant="caption" color="text.secondary">
                        去背景: {(comparison.details.alignment_optimization.timing.remove_background * 1000).toFixed(2)}ms
                      </Typography>
                    )}
                    {(() => {
                      const timing = comparison.details.alignment_optimization.timing
                      // 排除詳細步驟，只計算主要階段和總時間
                      const mainTimingKeys = [
                        'remove_background_total', 'remove_background',
                        'stage1_translation_coarse', 'stage2_rotation_coarse', 
                        'stage3_translation_fine', 'stage4_rotation_fine', 
                        'stage4_translation_fine', 'stage4_total'
                      ]
                      const totalTime = Object.entries(timing)
                        .filter(([key]) => mainTimingKeys.includes(key))
                        .reduce((sum, [, val]) => sum + (val || 0), 0)
                      return totalTime > 0 && (
                        <Typography variant="caption" color="primary" sx={{ fontWeight: 'bold', mt: 0.5 }}>
                          總時間: {(totalTime * 1000).toFixed(2)}ms
                        </Typography>
                      )
                    })()}
                  </Box>
                </Box>
              )}
            </Box>
          </Box>

          <Box sx={{ mb: 2, textAlign: 'center' }}>
            <Typography variant="caption" color="primary" sx={{ fontStyle: 'italic' }}>
              ↓ 以下視覺化圖表基於相同的校正後圖像生成 ↓
            </Typography>
          </Box>

          <VerificationView comparisonId={comparisonId} />
        </>
      )}

      {comparison.status === 'failed' && (
        <Alert 
          severity="error"
          action={
            <Button
              size="small"
              onClick={() => retryMutation.mutate()}
              disabled={retryMutation.isPending}
            >
              {retryMutation.isPending ? '重試中...' : '重試'}
            </Button>
          }
        >
          比對處理失敗
          {comparison.details?.error && (
            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
              錯誤: {comparison.details.error}
            </Typography>
          )}
        </Alert>
      )}
    </Paper>
  )
}

export default ComparisonResult



