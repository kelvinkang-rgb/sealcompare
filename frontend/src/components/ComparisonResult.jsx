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
import { useQuery } from '@tanstack/react-query'
import { comparisonAPI, visualizationAPI } from '../services/api'
import VerificationView from './VerificationView'
import MetricsExplanationDialog from './MetricsExplanationDialog'

function ComparisonResult({ comparisonId, onResetComparison }) {
  const [pollingInterval, setPollingInterval] = useState(1000)
  const [explanationOpen, setExplanationOpen] = useState(false)

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
          <Alert severity="info">
            處理中... {status?.progress}% - {status?.message}
          </Alert>
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
              {comparison.rotation_angle && (
                <Typography variant="body2">
                  旋轉角度: {comparison.rotation_angle}°
                </Typography>
              )}
              {comparison.improvement && (
                <Typography variant="body2">
                  改善幅度: {(comparison.improvement * 100).toFixed(2)}%
                </Typography>
              )}
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
                <Typography variant="body2">
                  SSIM: {(comparison.details.ssim * 100).toFixed(2)}%
                </Typography>
                <Typography variant="body2">
                  模板匹配: {(comparison.details.template_match * 100).toFixed(2)}%
                </Typography>
                <Typography variant="body2">
                  像素差異: {(comparison.details.pixel_diff * 100).toFixed(2)}%
                </Typography>
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
        <Alert severity="error">比對處理失敗</Alert>
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

