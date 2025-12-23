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



