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
  LinearProgress,
  List,
  ListItem,
  ListItemText,
} from '@mui/material'
import { Refresh as RefreshIcon, Info as InfoIcon, CheckCircle, RadioButtonUnchecked } from '@mui/icons-material'
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

  const isCompleted = comparison.status === 'completed'
  const isFailed = comparison.status === 'failed'

  return (
    <Paper sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Box>
          <Typography variant="h6">
            比對結果
          </Typography>
          {isCompleted && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
              比對已完成，點擊「重新比對」可開始新的比對
            </Typography>
          )}
        </Box>
        {onResetComparison && (
          <Button
            variant={isCompleted ? 'contained' : 'outlined'}
            color={isCompleted ? 'primary' : 'inherit'}
            startIcon={<RefreshIcon />}
            onClick={onResetComparison}
            disabled={isProcessing}
            sx={{ ml: 2 }}
          >
            重新比對
          </Button>
        )}
      </Box>

      {isProcessing && (
        <Box sx={{ mb: 2 }}>
          <Alert severity="info" icon={<CircularProgress size={20} />}>
            <Typography variant="body2" component="div" sx={{ mb: 1 }}>
              <strong>處理中...</strong> {status?.progress?.toFixed(0) || 0}%
            </Typography>
            {status?.progress !== undefined && (
              <LinearProgress 
                variant="determinate" 
                value={status.progress} 
                sx={{ mb: 1, height: 8, borderRadius: 4 }}
              />
            )}
            <Typography variant="body2" component="div" sx={{ mb: 1 }}>
              {status?.message || '正在處理中...'}
            </Typography>
            {status?.current_step && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                當前步驟: {status.current_step}
              </Typography>
            )}
            
            {/* 步驟列表 */}
            <Box sx={{ mt: 2 }}>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                處理步驟:
              </Typography>
              <List dense sx={{ py: 0 }}>
                {[
                  { step: '讀取圖像', progress: 10 },
                  { step: '圖像預處理', progress: 20 },
                  { step: '旋轉搜索/圖像比對', progress: 30 },
                  { step: '保存結果', progress: 60 },
                  { step: '計算對齊指標', progress: 70 },
                  { step: '生成視覺化', progress: 80 },
                  { step: '完成', progress: 100 },
                ].map((item, index) => {
                  const isCompleted = status?.progress !== undefined && status.progress >= item.progress
                  const isCurrent = status?.current_step && 
                    (item.step.includes(status.current_step) || 
                     status.current_step.includes(item.step))
                  
                  return (
                    <ListItem key={index} sx={{ py: 0.5, px: 0 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                        {isCompleted ? (
                          <CheckCircle sx={{ fontSize: 16, color: 'success.main', mr: 1 }} />
                        ) : isCurrent ? (
                          <CircularProgress size={16} sx={{ mr: 1 }} />
                        ) : (
                          <RadioButtonUnchecked sx={{ fontSize: 16, color: 'text.disabled', mr: 1 }} />
                        )}
                        <ListItemText
                          primary={
                            <Typography 
                              variant="caption" 
                              sx={{ 
                                color: isCurrent ? 'primary.main' : 'text.secondary',
                                fontWeight: isCurrent ? 'bold' : 'normal'
                              }}
                            >
                              {item.step}
                            </Typography>
                          }
                        />
                      </Box>
                    </ListItem>
                  )
                })}
              </List>
            </Box>
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

      {isFailed && (
        <Box>
          <Alert severity="error" sx={{ mb: 2 }}>
            <Typography variant="body2" component="div">
              <strong>比對處理失敗</strong>
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
              請檢查圖像文件是否正確，或點擊「重新比對」嘗試使用其他圖像。
            </Typography>
          </Alert>
          {onResetComparison && (
            <Box sx={{ textAlign: 'center', mt: 2 }}>
              <Button
                variant="contained"
                color="error"
                startIcon={<RefreshIcon />}
                onClick={onResetComparison}
              >
                重新比對
              </Button>
            </Box>
          )}
        </Box>
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

