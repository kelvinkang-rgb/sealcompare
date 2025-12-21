import React, { useState, useEffect, useRef } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import {
  Container,
  Typography,
  Button,
  Box,
  Paper,
  TextField,
  Alert,
  CircularProgress,
  Grid,
} from '@mui/material'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { comparisonAPI, imageAPI } from '../services/api'
import ComparisonForm from '../components/ComparisonForm'
import ComparisonResult from '../components/ComparisonResult'

function Comparison() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [searchParams] = useSearchParams()
  const [comparisonId, setComparisonId] = useState(null)
  const [threshold, setThreshold] = useState(0.95)
  const [isViewMode, setIsViewMode] = useState(false)
  const formRef = useRef(null)

  // 從 URL 參數讀取 comparisonId（查看已有記錄）
  useEffect(() => {
    const urlComparisonId = searchParams.get('comparisonId')
    if (urlComparisonId) {
      setComparisonId(urlComparisonId)
      setIsViewMode(true)
    }
  }, [searchParams])

  const createComparisonMutation = useMutation({
    mutationFn: async (data) => {
      return await comparisonAPI.create(data)
    },
    onSuccess: (data) => {
      setComparisonId(data.id)
      queryClient.invalidateQueries(['comparisons'])
    },
  })

  const handleSubmit = async (image1Id, image2Id, enableRotation) => {
    createComparisonMutation.mutate({
      image1_id: image1Id,
      image2_id: image2Id,
      threshold: threshold,
      enable_rotation_search: enableRotation,
      enable_translation_search: true,  // 預設開啟，因為人工標記印鑑無法確保中心點都一致
    })
  }

  const handleResetComparison = () => {
    // 重置比對狀態，重新顯示表單
    setComparisonId(null)
    createComparisonMutation.reset()
    // 重置表單狀態（包括已上傳的圖像）
    if (formRef.current) {
      formRef.current.reset()
    }
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Button onClick={() => navigate('/')} sx={{ mb: 2 }}>
          返回首頁
        </Button>
        {isViewMode && (
          <Button onClick={() => navigate('/history')} sx={{ mb: 2, ml: 2 }}>
            返回記錄列表
          </Button>
        )}
        <Typography variant="h4" component="h1" gutterBottom>
          {isViewMode ? '查看比對結果' : '圖像比對'}
        </Typography>
      </Box>

      <Grid container spacing={3}>
        {/* 只有在非查看模式且沒有進行中的比對時才顯示上傳表單 */}
        {!isViewMode && !comparisonId && (
          <>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  上傳圖像
                </Typography>
                <ComparisonForm ref={formRef} onSubmit={handleSubmit} />
              </Paper>
            </Grid>

            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  比對設置
                </Typography>
                <TextField
                  label="相似度閾值"
                  type="number"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  inputProps={{ min: 0, max: 1, step: 0.01 }}
                  fullWidth
                  sx={{ mb: 2 }}
                />
              </Paper>
            </Grid>

            {createComparisonMutation.isError && (
              <Grid item xs={12}>
                <Alert severity="error">
                  {createComparisonMutation.error?.response?.data?.detail ||
                    '創建比對失敗'}
                </Alert>
              </Grid>
            )}
          </>
        )}

        {/* 顯示比對進行中的提示 */}
        {!isViewMode && createComparisonMutation.isPending && !comparisonId && (
          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
              <CircularProgress />
            </Box>
          </Grid>
        )}

        {/* 顯示比對結果 */}
        {comparisonId && (
          <Grid item xs={12}>
            <ComparisonResult 
              comparisonId={comparisonId} 
              onResetComparison={!isViewMode ? handleResetComparison : undefined}
            />
          </Grid>
        )}

        {/* 如果沒有 comparisonId 且是查看模式，顯示錯誤 */}
        {isViewMode && !comparisonId && (
          <Grid item xs={12}>
            <Alert severity="warning">
              無法載入比對記錄，請檢查記錄 ID 是否正確
            </Alert>
          </Grid>
        )}
      </Grid>
    </Container>
  )
}

export default Comparison

