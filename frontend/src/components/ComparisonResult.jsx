import React, { useEffect, useState } from 'react'
import {
  Box,
  Paper,
  Typography,
  CircularProgress,
  Alert,
  Grid,
  Chip,
  Button,
  IconButton,
} from '@mui/material'
import { Refresh as RefreshIcon, Info as InfoIcon } from '@mui/icons-material'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { comparisonAPI, visualizationAPI } from '../services/api'
import VerificationView from './VerificationView'
import MetricsExplanationDialog from './MetricsExplanationDialog'
import ProcessingStages from './ProcessingStages'

function ComparisonResult({ comparisonId, onResetComparison }) {
  const [pollingInterval, setPollingInterval] = useState(1000)
  const [explanationOpen, setExplanationOpen] = useState(false)
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
            處理中... {status?.progress}% - {status?.message}
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
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={12} md={6}>
              <Typography variant="body1">
                相似度: <strong>{(comparison.similarity * 100).toFixed(2)}%</strong>
              </Typography>
              <Typography variant="body1">
                閾值: <strong>{(comparison.threshold * 100).toFixed(2)}%</strong>
              </Typography>
              <Chip
                label={comparison.is_match ? '✓ 匹配' : '✗ 不匹配'}
                color={comparison.is_match ? 'success' : 'error'}
                sx={{ mt: 1 }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
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
              </Box>
            </Grid>
          </Grid>

          {comparison.details && (
            <Box sx={{ mb: 2, p: 2, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Typography variant="subtitle2">
                  詳細指標（校正後）
                </Typography>
                <IconButton
                  size="small"
                  onClick={() => setExplanationOpen(true)}
                  sx={{ color: 'primary.main' }}
                  title="查看指標說明"
                >
                  <InfoIcon fontSize="small" />
                </IconButton>
              </Box>
              <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                以下指標基於校正後的圖像計算，與下方校正驗證圖表完全對應
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                {comparison.details.ssim !== null && comparison.details.ssim !== undefined && (
                  <Typography variant="body2">
                    SSIM: <strong>{(comparison.details.ssim * 100).toFixed(2)}%</strong>
                  </Typography>
                )}
                {comparison.details.template_match !== null && comparison.details.template_match !== undefined && (
                  <Typography variant="body2">
                    模板匹配: <strong>{(comparison.details.template_match * 100).toFixed(2)}%</strong>
                  </Typography>
                )}
                {comparison.details.pixel_diff !== null && comparison.details.pixel_diff !== undefined && (
                  <Typography variant="body2">
                    像素差異: <strong>{(comparison.details.pixel_diff * 100).toFixed(2)}%</strong>
                  </Typography>
                )}
                {comparison.details.pixel_similarity !== null && comparison.details.pixel_similarity !== undefined && (
                  <Typography variant="body2">
                    像素相似度: <strong>{(comparison.details.pixel_similarity * 100).toFixed(2)}%</strong>
                  </Typography>
                )}
                {comparison.details.histogram_similarity !== null && comparison.details.histogram_similarity !== undefined && (
                  <Typography variant="body2">
                    直方圖相似度: <strong>{(comparison.details.histogram_similarity * 100).toFixed(2)}%</strong>
                  </Typography>
                )}
                {comparison.details.rotation_angle !== null && comparison.details.rotation_angle !== undefined && comparison.details.rotation_angle !== 0 && (
                  <Typography variant="body2">
                    旋轉角度: <strong>{comparison.details.rotation_angle.toFixed(2)}°</strong>
                  </Typography>
                )}
                {comparison.details.translation_offset && (
                  <Typography variant="body2">
                    平移偏移: <strong>X: {comparison.details.translation_offset.x}px, Y: {comparison.details.translation_offset.y}px</strong>
                  </Typography>
                )}
              </Box>
            </Box>
          )}

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

      {/* 指標說明對話框 */}
      <MetricsExplanationDialog
        open={explanationOpen}
        onClose={() => setExplanationOpen(false)}
      />
    </Paper>
  )
}

export default ComparisonResult



