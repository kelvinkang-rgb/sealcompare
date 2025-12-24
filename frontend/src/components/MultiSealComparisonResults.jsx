import React, { useState } from 'react'
import {
  Box,
  Paper,
  Typography,
  Grid,
  Chip,
  CircularProgress,
  Alert,
  Card,
  CardContent,
} from '@mui/material'
import { CheckCircle as CheckCircleIcon, Cancel as CancelIcon } from '@mui/icons-material'
import ImagePreviewDialog from './ImagePreviewDialog'

function MultiSealComparisonResults({ results, image1Id }) {
  const [modalOpen, setModalOpen] = useState(false)
  const [modalImageUrl, setModalImageUrl] = useState('')
  const [modalImage, setModalImage] = useState(null)

  const handleImageClick = (imagePath, title) => {
    if (!imagePath) return
    
    // 構建完整的圖片 URL
    const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 
      (import.meta.env.PROD ? '/api/v1' : 'http://localhost:8000/api/v1')
    
    // 提取文件名
    const fileName = imagePath.split('/').pop()
    const imageUrl = `${API_BASE_URL}/images/multi-seal-comparisons/${fileName}`
    
    // 構建一個簡單的 image 對象（用於顯示 filename）
    const image = {
      id: imagePath, // 使用路徑作為唯一標識
      filename: fileName || title || '圖片預覽'
    }
    
    setModalImageUrl(imageUrl)
    setModalImage(image)
    setModalOpen(true)
  }
  
  const getImageUrl = (imagePath) => {
    if (!imagePath) return null
    
    const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 
      (import.meta.env.PROD ? '/api/v1' : 'http://localhost:8000/api/v1')
    
    const fileName = imagePath.split('/').pop()
    return `${API_BASE_URL}/images/multi-seal-comparisons/${fileName}`
  }

  const handleCloseModal = () => {
    setModalOpen(false)
    setModalImageUrl('')
    setModalImage(null)
  }

  if (!results || results.length === 0) {
    return (
      <Alert severity="info" sx={{ mt: 2 }}>
        暫無比對結果
      </Alert>
    )
  }

  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h6" gutterBottom>
        比對結果 ({results.length} 個印鑑)
      </Typography>
      
      <Grid container spacing={2}>
        {results.map((result, index) => (
          <Grid item xs={12} key={result.seal_image_id || index}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2, gap: 2 }}>
                  <Typography variant="h6" component="span">
                    印鑑 {result.seal_index}
                  </Typography>
                  
                  {result.error ? (
                    <Chip
                      icon={<CancelIcon />}
                      label="比對失敗"
                      color="error"
                      size="small"
                    />
                  ) : result.similarity !== null ? (
                    <>
                      <Chip
                        icon={result.is_match ? <CheckCircleIcon /> : <CancelIcon />}
                        label={result.is_match ? '匹配' : '不匹配'}
                        color={result.is_match ? 'success' : 'error'}
                        size="small"
                      />
                      <Chip
                        label={`相似度: ${(result.similarity * 100).toFixed(2)}%`}
                        color={result.is_match ? 'success' : 'default'}
                        size="small"
                        variant="outlined"
                      />
                    </>
                  ) : (
                    <CircularProgress size={20} />
                  )}
                </Box>

                {result.error && (
                  <Alert severity="error" sx={{ mb: 2 }}>
                    {result.error}
                  </Alert>
                )}

                {!result.error && (
                  <Grid container spacing={1}>
                    {/* 輸入圖像1：去背景後的圖像1 */}
                    <Grid item xs={12} sm={6} md={2} sx={{ flexBasis: { md: '20%' }, maxWidth: { md: '20%' } }}>
                      <Paper
                        sx={{
                          p: 1,
                          textAlign: 'center',
                          cursor: result.input_image1_path ? 'pointer' : 'default',
                          '&:hover': result.input_image1_path ? { opacity: 0.8 } : {},
                        }}
                        onClick={() => result.input_image1_path && handleImageClick(
                          result.input_image1_path,
                          `印鑑 ${result.seal_index} - 輸入圖像1: 去背景後的圖像1`
                        )}
                      >
                        <Typography variant="caption" display="block" gutterBottom>
                          輸入圖像1: 去背景後的圖像1
                        </Typography>
                        {result.input_image1_path ? (
                          <img
                            src={getImageUrl(result.input_image1_path)}
                            alt="輸入圖像1"
                            style={{
                              maxWidth: '100%',
                              height: 'auto',
                              borderRadius: '4px',
                            }}
                            onError={(e) => {
                              e.target.style.display = 'none'
                              e.target.parentElement.innerHTML =
                                '<p style="color: #999; text-align: center; padding: 20px;">圖片載入失敗</p>'
                            }}
                          />
                        ) : (
                          <Box sx={{ p: 2, color: 'text.secondary' }}>
                            未生成
                          </Box>
                        )}
                      </Paper>
                    </Grid>

                    {/* 輸入圖像2：對齊後的印鑑圖像 */}
                    <Grid item xs={12} sm={6} md={2} sx={{ flexBasis: { md: '20%' }, maxWidth: { md: '20%' } }}>
                      <Paper
                        sx={{
                          p: 1,
                          textAlign: 'center',
                          cursor: result.input_image2_path ? 'pointer' : 'default',
                          '&:hover': result.input_image2_path ? { opacity: 0.8 } : {},
                        }}
                        onClick={() => result.input_image2_path && handleImageClick(
                          result.input_image2_path,
                          `印鑑 ${result.seal_index} - 輸入圖像2: 對齊後的印鑑圖像`
                        )}
                      >
                        <Typography variant="caption" display="block" gutterBottom>
                          輸入圖像2: 對齊後的印鑑圖像
                        </Typography>
                        {result.input_image2_path ? (
                          <img
                            src={getImageUrl(result.input_image2_path)}
                            alt="輸入圖像2"
                            style={{
                              maxWidth: '100%',
                              height: 'auto',
                              borderRadius: '4px',
                            }}
                            onError={(e) => {
                              e.target.style.display = 'none'
                              e.target.parentElement.innerHTML =
                                '<p style="color: #999; text-align: center; padding: 20px;">圖片載入失敗</p>'
                            }}
                          />
                        ) : (
                          <Box sx={{ p: 2, color: 'text.secondary' }}>
                            未生成
                          </Box>
                        )}
                      </Paper>
                    </Grid>

                    {/* 疊圖1 */}
                    <Grid item xs={12} sm={6} md={2} sx={{ flexBasis: { md: '20%' }, maxWidth: { md: '20%' } }}>
                      <Paper
                        sx={{
                          p: 1,
                          textAlign: 'center',
                          cursor: result.overlay1_path ? 'pointer' : 'default',
                          '&:hover': result.overlay1_path ? { opacity: 0.8 } : {},
                        }}
                        onClick={() => result.overlay1_path && handleImageClick(
                          result.overlay1_path,
                          `印鑑 ${result.seal_index} - 疊圖1: 圖像1疊在印鑑上`
                        )}
                      >
                        <Typography variant="caption" display="block" gutterBottom>
                          疊圖1: 圖像1疊在印鑑上
                        </Typography>
                        {result.overlay1_path ? (
                          <img
                            src={getImageUrl(result.overlay1_path)}
                            alt="疊圖1"
                            style={{
                              width: '100%',
                              height: '200px',
                              objectFit: 'contain',
                              borderRadius: '4px',
                            }}
                            onError={(e) => {
                              e.target.style.display = 'none'
                              e.target.parentElement.innerHTML =
                                '<p style="color: #999; text-align: center; padding: 20px;">圖片載入失敗</p>'
                            }}
                          />
                        ) : (
                          <Box sx={{ p: 2, color: 'text.secondary' }}>
                            未生成
                          </Box>
                        )}
                      </Paper>
                    </Grid>

                    {/* 疊圖2 */}
                    <Grid item xs={12} sm={6} md={2} sx={{ flexBasis: { md: '20%' }, maxWidth: { md: '20%' } }}>
                      <Paper
                        sx={{
                          p: 1,
                          textAlign: 'center',
                          cursor: result.overlay2_path ? 'pointer' : 'default',
                          '&:hover': result.overlay2_path ? { opacity: 0.8 } : {},
                        }}
                        onClick={() => result.overlay2_path && handleImageClick(
                          result.overlay2_path,
                          `印鑑 ${result.seal_index} - 疊圖2: 印鑑疊在圖像1上`
                        )}
                      >
                        <Typography variant="caption" display="block" gutterBottom>
                          疊圖2: 印鑑疊在圖像1上
                        </Typography>
                        {result.overlay2_path ? (
                          <img
                            src={getImageUrl(result.overlay2_path)}
                            alt="疊圖2"
                            style={{
                              width: '100%',
                              height: '200px',
                              objectFit: 'contain',
                              borderRadius: '4px',
                            }}
                            onError={(e) => {
                              e.target.style.display = 'none'
                              e.target.parentElement.innerHTML =
                                '<p style="color: #999; text-align: center; padding: 20px;">圖片載入失敗</p>'
                            }}
                          />
                        ) : (
                          <Box sx={{ p: 2, color: 'text.secondary' }}>
                            未生成
                          </Box>
                        )}
                      </Paper>
                    </Grid>

                    {/* 熱力圖 */}
                    <Grid item xs={12} sm={6} md={2} sx={{ flexBasis: { md: '20%' }, maxWidth: { md: '20%' } }}>
                      <Paper
                        sx={{
                          p: 1,
                          textAlign: 'center',
                          cursor: result.heatmap_path ? 'pointer' : 'default',
                          '&:hover': result.heatmap_path ? { opacity: 0.8 } : {},
                        }}
                        onClick={() => result.heatmap_path && handleImageClick(
                          result.heatmap_path,
                          `印鑑 ${result.seal_index} - 差異熱力圖`
                        )}
                      >
                        <Typography variant="caption" display="block" gutterBottom>
                          差異熱力圖
                        </Typography>
                        {result.heatmap_path ? (
                          <img
                            src={getImageUrl(result.heatmap_path)}
                            alt="熱力圖"
                            style={{
                              width: '100%',
                              height: '200px',
                              objectFit: 'contain',
                              borderRadius: '4px',
                            }}
                            onError={(e) => {
                              e.target.style.display = 'none'
                              e.target.parentElement.innerHTML =
                                '<p style="color: #999; text-align: center; padding: 20px;">圖片載入失敗</p>'
                            }}
                          />
                        ) : (
                          <Box sx={{ p: 2, color: 'text.secondary' }}>
                            未生成
                          </Box>
                        )}
                      </Paper>
                    </Grid>
                  </Grid>
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <ImagePreviewDialog
        open={modalOpen}
        onClose={handleCloseModal}
        image={modalImage}
        imageUrl={modalImageUrl}
        sealBbox={null}
        seals={[]}
      />
    </Box>
  )
}

export default MultiSealComparisonResults

