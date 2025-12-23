import React, { useState } from 'react'
import { Box, Typography, Grid, Paper } from '@mui/material'
import { visualizationAPI } from '../services/api'
import ImageModal from './ImageModal'

function VerificationView({ comparisonId }) {
  const [modalOpen, setModalOpen] = useState(false)
  const [modalImageUrl, setModalImageUrl] = useState('')
  const [modalTitle, setModalTitle] = useState('')

  const handleImageClick = (imageUrl, title) => {
    setModalImageUrl(imageUrl)
    setModalTitle(title)
    setModalOpen(true)
  }

  const handleCloseModal = () => {
    setModalOpen(false)
    setModalImageUrl('')
    setModalTitle('')
  }

  return (
    <Box sx={{ mt: 3 }}>
      <Typography variant="h6" gutterBottom>
        校正驗證
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="subtitle2" gutterBottom>
              並排對比
            </Typography>
            <Box
              onClick={() =>
                handleImageClick(
                  visualizationAPI.getComparisonImage(comparisonId),
                  '並排對比圖'
                )
              }
              sx={{
                cursor: 'pointer',
                '&:hover': {
                  opacity: 0.8,
                },
              }}
            >
              <img
                src={visualizationAPI.getComparisonImage(comparisonId)}
                alt="並排對比"
                style={{ 
                  maxWidth: '100%', 
                  maxHeight: '400px',
                  width: 'auto',
                  height: 'auto', 
                  borderRadius: '4px',
                  objectFit: 'contain'
                }}
                onError={(e) => {
                  e.target.style.display = 'none'
                  e.target.parentElement.innerHTML =
                    '<p style="color: #999; text-align: center;">圖片載入失敗</p>'
                }}
              />
            </Box>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              點擊圖片放大查看
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="subtitle2" gutterBottom>
              差異熱力圖
            </Typography>
            <Box
              onClick={() =>
                handleImageClick(visualizationAPI.getHeatmap(comparisonId), '差異熱力圖')
              }
              sx={{
                cursor: 'pointer',
                '&:hover': {
                  opacity: 0.8,
                },
              }}
            >
              <img
                src={visualizationAPI.getHeatmap(comparisonId)}
                alt="差異熱力圖"
                style={{ 
                  maxWidth: '100%', 
                  maxHeight: '400px',
                  width: 'auto',
                  height: 'auto', 
                  borderRadius: '4px',
                  objectFit: 'contain'
                }}
                onError={(e) => {
                  e.target.style.display = 'none'
                  e.target.parentElement.innerHTML =
                    '<p style="color: #999; text-align: center;">圖片載入失敗</p>'
                }}
              />
            </Box>
            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
              點擊圖片放大查看
            </Typography>
          </Paper>
        </Grid>
        
        {/* 疊圖比對區域 */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom sx={{ textAlign: 'center', mb: 2 }}>
              疊圖比對
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1, fontWeight: 'medium', display: 'block' }}>
                    圖像1疊在圖像2校正上
                  </Typography>
                  <Box
                    onClick={() =>
                      handleImageClick(
                        visualizationAPI.getOverlay(comparisonId, '1'),
                        '疊圖1: 圖像1疊在圖像2校正上'
                      )
                    }
                    sx={{
                      cursor: 'pointer',
                      '&:hover': {
                        opacity: 0.8,
                      },
                    }}
                  >
                    <img
                      src={visualizationAPI.getOverlay(comparisonId, '1')}
                      alt="疊圖1"
                      style={{ 
                        maxWidth: '100%', 
                        maxHeight: '300px',
                        width: 'auto',
                        height: 'auto', 
                        borderRadius: '4px',
                        objectFit: 'contain'
                      }}
                      onError={(e) => {
                        e.target.style.display = 'none'
                        e.target.parentElement.innerHTML =
                          '<p style="color: #999; text-align: center;">疊圖載入失敗</p>'
                      }}
                    />
                  </Box>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1.5, display: 'block', lineHeight: 1.6 }}>
                    <Box component="span" sx={{ display: 'block', mb: 0.5 }}>
                      <strong>顏色說明：</strong>
                    </Box>
                    <Box component="span" sx={{ display: 'block' }}>
                      <span style={{ color: '#000' }}>■</span> 黑色：獨有區域差異
                    </Box>
                    <Box component="span" sx={{ display: 'block' }}>
                      <span style={{ color: '#FF8C00' }}>■</span> 橘色：重疊區域像素差異
                    </Box>
                    <Box component="span" sx={{ display: 'block', mt: 0.5, fontSize: '0.875rem' }}>
                      點擊圖片放大查看
                    </Box>
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={6}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1, fontWeight: 'medium', display: 'block' }}>
                    圖像2校正疊在圖像1上
                  </Typography>
                  <Box
                    onClick={() =>
                      handleImageClick(
                        visualizationAPI.getOverlay(comparisonId, '2'),
                        '疊圖2: 圖像2校正疊在圖像1上'
                      )
                    }
                    sx={{
                      cursor: 'pointer',
                      '&:hover': {
                        opacity: 0.8,
                      },
                    }}
                  >
                    <img
                      src={visualizationAPI.getOverlay(comparisonId, '2')}
                      alt="疊圖2"
                      style={{ 
                        maxWidth: '100%', 
                        maxHeight: '300px',
                        width: 'auto',
                        height: 'auto', 
                        borderRadius: '4px',
                        objectFit: 'contain'
                      }}
                      onError={(e) => {
                        e.target.style.display = 'none'
                        e.target.parentElement.innerHTML =
                          '<p style="color: #999; text-align: center;">疊圖載入失敗</p>'
                      }}
                    />
                  </Box>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1.5, display: 'block', lineHeight: 1.6 }}>
                    <Box component="span" sx={{ display: 'block', mb: 0.5 }}>
                      <strong>顏色說明：</strong>
                    </Box>
                    <Box component="span" sx={{ display: 'block' }}>
                      <span style={{ color: '#000' }}>■</span> 黑色：獨有區域差異
                    </Box>
                    <Box component="span" sx={{ display: 'block' }}>
                      <span style={{ color: '#FF8C00' }}>■</span> 橘色：重疊區域像素差異
                    </Box>
                    <Box component="span" sx={{ display: 'block', mt: 0.5, fontSize: '0.875rem' }}>
                      點擊圖片放大查看
                    </Box>
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>

      <ImageModal
        open={modalOpen}
        onClose={handleCloseModal}
        imageUrl={modalImageUrl}
        title={modalTitle}
      />
    </Box>
  )
}

export default VerificationView

