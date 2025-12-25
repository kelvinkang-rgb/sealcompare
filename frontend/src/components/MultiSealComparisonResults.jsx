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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableRow,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material'
import { CheckCircle as CheckCircleIcon, Cancel as CancelIcon, ExpandMore as ExpandMoreIcon } from '@mui/icons-material'
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
                  ) : result.mask_based_similarity !== null && result.mask_based_similarity !== undefined ? (
                    <>
                      <Chip
                        icon={result.is_match ? <CheckCircleIcon /> : <CancelIcon />}
                        label={result.is_match ? '匹配' : '不匹配'}
                        color={result.is_match ? 'success' : 'error'}
                        size="small"
                      />
                      <Chip
                        label={`Mask相似度: ${(result.mask_based_similarity * 100).toFixed(2)}%`}
                        color={result.is_match ? 'success' : 'info'}
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
                    <Grid item xs={12} sm={6} md={3} sx={{ flexBasis: { md: '25%' }, maxWidth: { md: '25%' } }}>
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
                    <Grid item xs={12} sm={6} md={3} sx={{ flexBasis: { md: '25%' }, maxWidth: { md: '25%' } }}>
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
                    <Grid item xs={12} sm={6} md={3} sx={{ flexBasis: { md: '25%' }, maxWidth: { md: '25%' } }}>
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
                              console.error(`疊圖1載入失敗 (印鑑 ${result.seal_index}):`, result.overlay1_path, getImageUrl(result.overlay1_path))
                              e.target.style.display = 'none'
                              const errorMsg = e.target.parentElement.querySelector('.error-message') || document.createElement('p')
                              errorMsg.className = 'error-message'
                              errorMsg.style.cssText = 'color: #f44336; text-align: center; padding: 20px; font-size: 0.75rem;'
                              errorMsg.textContent = `圖片載入失敗: ${result.overlay1_path}`
                              if (!e.target.parentElement.querySelector('.error-message')) {
                                e.target.parentElement.appendChild(errorMsg)
                              }
                            }}
                            onLoad={() => {
                              console.log(`疊圖1載入成功 (印鑑 ${result.seal_index}):`, result.overlay1_path)
                            }}
                          />
                        ) : (
                          <Box sx={{ p: 2, color: 'text.secondary' }}>
                            {result.overlay_error ? '生成失敗' : '未生成'}
                          </Box>
                        )}
                      </Paper>
                    </Grid>

                    {/* 疊圖2 */}
                    <Grid item xs={12} sm={6} md={3} sx={{ flexBasis: { md: '25%' }, maxWidth: { md: '25%' } }}>
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
                              console.error(`疊圖2載入失敗 (印鑑 ${result.seal_index}):`, result.overlay2_path, getImageUrl(result.overlay2_path))
                              e.target.style.display = 'none'
                              const errorMsg = e.target.parentElement.querySelector('.error-message') || document.createElement('p')
                              errorMsg.className = 'error-message'
                              errorMsg.style.cssText = 'color: #f44336; text-align: center; padding: 20px; font-size: 0.75rem;'
                              errorMsg.textContent = `圖片載入失敗: ${result.overlay2_path}`
                              if (!e.target.parentElement.querySelector('.error-message')) {
                                e.target.parentElement.appendChild(errorMsg)
                              }
                            }}
                            onLoad={() => {
                              console.log(`疊圖2載入成功 (印鑑 ${result.seal_index}):`, result.overlay2_path)
                            }}
                          />
                        ) : (
                          <Box sx={{ p: 2, color: 'text.secondary' }}>
                            {result.overlay_error ? '生成失敗' : '未生成'}
                          </Box>
                        )}
                      </Paper>
                    </Grid>
                  </Grid>
                )}

                {/* Mask像素差異視覺化區域 */}
                {!result.error && (
                  <Box sx={{ mt: 2 }}>
                    <Accordion defaultExpanded={false}>
                      <AccordionSummary
                        expandIcon={<ExpandMoreIcon />}
                        aria-controls="mask-visualization-content"
                        id="mask-visualization-header"
                      >
                        <Typography variant="subtitle1" sx={{ fontWeight: 'medium' }}>
                          Mask像素差異視覺化
                        </Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Grid container spacing={1}>
                          {/* 重疊區域mask */}
                          <Grid item xs={12} sm={6} md={3}>
                            <Paper
                              sx={{
                                p: 1,
                                textAlign: 'center',
                                cursor: result.overlap_mask_path ? 'pointer' : 'default',
                                '&:hover': result.overlap_mask_path ? { opacity: 0.8 } : {},
                              }}
                              onClick={() => result.overlap_mask_path && handleImageClick(
                                result.overlap_mask_path,
                                `印鑑 ${result.seal_index} - 重疊區域 (Overlap Mask)`
                              )}
                            >
                              <Typography variant="caption" display="block" gutterBottom>
                                重疊區域 (Overlap Mask)
                              </Typography>
                              {result.overlap_mask_path ? (
                                <Box
                                  sx={{
                                    width: '100%',
                                    height: '280px', // 統一高度以匹配 gray_diff
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    backgroundColor: '#f5f5f5',
                                    borderRadius: '4px',
                                  }}
                                >
                                  <img
                                    src={getImageUrl(result.overlap_mask_path)}
                                    alt="重疊區域mask"
                                    style={{
                                      maxWidth: '100%',
                                      maxHeight: '100%',
                                      width: 'auto',
                                      height: 'auto',
                                      objectFit: 'contain',
                                      borderRadius: '4px',
                                    }}
                                    onError={(e) => {
                                      e.target.style.display = 'none'
                                      e.target.parentElement.innerHTML =
                                        '<p style="color: #999; text-align: center; padding: 20px;">圖片載入失敗</p>'
                                    }}
                                  />
                                </Box>
                              ) : (
                                <Box sx={{ p: 2, color: 'text.secondary' }}>
                                  未生成
                                </Box>
                              )}
                            </Paper>
                          </Grid>

                          {/* 像素差異mask */}
                          <Grid item xs={12} sm={6} md={3}>
                            <Paper
                              sx={{
                                p: 1,
                                textAlign: 'center',
                                cursor: result.pixel_diff_mask_path ? 'pointer' : 'default',
                                '&:hover': result.pixel_diff_mask_path ? { opacity: 0.8 } : {},
                              }}
                              onClick={() => result.pixel_diff_mask_path && handleImageClick(
                                result.pixel_diff_mask_path,
                                `印鑑 ${result.seal_index} - 像素差異 (Pixel Difference Mask)`
                              )}
                            >
                              <Typography variant="caption" display="block" gutterBottom>
                                像素差異 (Pixel Difference Mask)
                              </Typography>
                              {result.pixel_diff_mask_path ? (
                                <Box
                                  sx={{
                                    width: '100%',
                                    height: '280px', // 統一高度以匹配 gray_diff
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    backgroundColor: '#f5f5f5',
                                    borderRadius: '4px',
                                  }}
                                >
                                  <img
                                    src={getImageUrl(result.pixel_diff_mask_path)}
                                    alt="像素差異mask"
                                    style={{
                                      maxWidth: '100%',
                                      maxHeight: '100%',
                                      width: 'auto',
                                      height: 'auto',
                                      objectFit: 'contain',
                                      borderRadius: '4px',
                                    }}
                                    onError={(e) => {
                                      e.target.style.display = 'none'
                                      e.target.parentElement.innerHTML =
                                        '<p style="color: #999; text-align: center; padding: 20px;">圖片載入失敗</p>'
                                    }}
                                  />
                                </Box>
                              ) : (
                                <Box sx={{ p: 2, color: 'text.secondary' }}>
                                  未生成
                                </Box>
                              )}
                            </Paper>
                          </Grid>

                          {/* 圖像2獨有區域mask */}
                          <Grid item xs={12} sm={6} md={3}>
                            <Paper
                              sx={{
                                p: 1,
                                textAlign: 'center',
                                cursor: result.diff_mask_2_only_path ? 'pointer' : 'default',
                                '&:hover': result.diff_mask_2_only_path ? { opacity: 0.8 } : {},
                              }}
                              onClick={() => result.diff_mask_2_only_path && handleImageClick(
                                result.diff_mask_2_only_path,
                                `印鑑 ${result.seal_index} - 圖像2獨有區域 (Image 2 Only)`
                              )}
                            >
                              <Typography variant="caption" display="block" gutterBottom>
                                圖像2獨有區域 (Image 2 Only)
                              </Typography>
                              {result.diff_mask_2_only_path ? (
                                <Box
                                  sx={{
                                    width: '100%',
                                    height: '280px', // 統一高度以匹配 gray_diff
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    backgroundColor: '#f5f5f5',
                                    borderRadius: '4px',
                                  }}
                                >
                                  <img
                                    src={getImageUrl(result.diff_mask_2_only_path)}
                                    alt="圖像2獨有區域mask"
                                    style={{
                                      maxWidth: '100%',
                                      maxHeight: '100%',
                                      width: 'auto',
                                      height: 'auto',
                                      objectFit: 'contain',
                                      borderRadius: '4px',
                                    }}
                                    onError={(e) => {
                                      e.target.style.display = 'none'
                                      e.target.parentElement.innerHTML =
                                        '<p style="color: #999; text-align: center; padding: 20px;">圖片載入失敗</p>'
                                    }}
                                  />
                                </Box>
                              ) : (
                                <Box sx={{ p: 2, color: 'text.secondary' }}>
                                  未生成
                          </Box>
                        )}
                      </Paper>
                    </Grid>

                          {/* 圖像1獨有區域mask */}
                          <Grid item xs={12} sm={6} md={3}>
                      <Paper
                        sx={{
                          p: 1,
                          textAlign: 'center',
                                cursor: result.diff_mask_1_only_path ? 'pointer' : 'default',
                                '&:hover': result.diff_mask_1_only_path ? { opacity: 0.8 } : {},
                        }}
                              onClick={() => result.diff_mask_1_only_path && handleImageClick(
                                result.diff_mask_1_only_path,
                                `印鑑 ${result.seal_index} - 圖像1獨有區域 (Image 1 Only)`
                        )}
                      >
                        <Typography variant="caption" display="block" gutterBottom>
                                圖像1獨有區域 (Image 1 Only)
                        </Typography>
                              {result.diff_mask_1_only_path ? (
                                <Box
                                  sx={{
                                    width: '100%',
                                    height: '280px', // 統一高度以匹配 gray_diff
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    backgroundColor: '#f5f5f5',
                                    borderRadius: '4px',
                                  }}
                                >
                                  <img
                                    src={getImageUrl(result.diff_mask_1_only_path)}
                                    alt="圖像1獨有區域mask"
                                    style={{
                                      maxWidth: '100%',
                                      maxHeight: '100%',
                                      width: 'auto',
                                      height: 'auto',
                                      objectFit: 'contain',
                                      borderRadius: '4px',
                                    }}
                                    onError={(e) => {
                                      e.target.style.display = 'none'
                                      e.target.parentElement.innerHTML =
                                        '<p style="color: #999; text-align: center; padding: 20px;">圖片載入失敗</p>'
                                    }}
                                  />
                                </Box>
                        ) : (
                          <Box sx={{ p: 2, color: 'text.secondary' }}>
                            未生成
                          </Box>
                        )}
                      </Paper>
                    </Grid>

                    {/* 灰度差異圖 (Gray Diff) */}
                    <Grid item xs={12} sm={6} md={3}>
                      <Paper
                        sx={{
                          p: 1,
                          textAlign: 'center',
                          cursor: result.gray_diff_path ? 'pointer' : 'default',
                          '&:hover': result.gray_diff_path ? { opacity: 0.8 } : {},
                        }}
                        onClick={() => result.gray_diff_path && handleImageClick(
                          result.gray_diff_path,
                          `印鑑 ${result.seal_index} - 灰度差異圖 (Gray Diff)`
                        )}
                      >
                        <Typography variant="caption" display="block" gutterBottom>
                          灰度差異圖 (Gray Diff)
                        </Typography>
                        {result.gray_diff_path ? (
                          <Box
                            sx={{
                              width: '100%',
                              height: '280px', // 為 legend 預留額外空間
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              backgroundColor: '#f5f5f5',
                              borderRadius: '4px',
                            }}
                          >
                            <img
                              src={getImageUrl(result.gray_diff_path)}
                              alt="灰度差異圖"
                              style={{
                                maxWidth: '100%',
                                maxHeight: '100%',
                                width: 'auto',
                                height: 'auto',
                                objectFit: 'contain',
                                borderRadius: '4px',
                              }}
                              onError={(e) => {
                                e.target.style.display = 'none'
                                e.target.parentElement.innerHTML =
                                  '<p style="color: #999; text-align: center; padding: 20px;">圖片載入失敗</p>'
                              }}
                            />
                          </Box>
                        ) : (
                          <Box sx={{ p: 2, color: 'text.secondary' }}>
                            未生成
                          </Box>
                        )}
                      </Paper>
                    </Grid>
                  </Grid>
                      </AccordionDetails>
                    </Accordion>
                  </Box>
                )}

                {/* Mask統計資訊區域 */}
                {!result.error && result.mask_statistics && (
                  <Box sx={{ mt: 2 }}>
                    <Accordion>
                      <AccordionSummary
                        expandIcon={<ExpandMoreIcon />}
                        aria-controls="mask-statistics-content"
                        id="mask-statistics-header"
                      >
                        <Typography variant="subtitle1" sx={{ fontWeight: 'medium' }}>
                          Mask統計資訊
                        </Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <TableContainer component={Paper} variant="outlined">
                          <Table size="small">
                            <TableBody>
                              {/* 像素數量組 */}
                              <TableRow>
                                <TableCell colSpan={2} sx={{ backgroundColor: 'grey.100', fontWeight: 'bold' }}>
                                  像素數量
                                </TableCell>
                              </TableRow>
                              <TableRow>
                                <TableCell sx={{ color: 'success.main', fontWeight: 'medium' }}>
                                  重疊區域像素
                                </TableCell>
                                <TableCell align="right">
                                  {result.mask_statistics.overlap_pixels?.toLocaleString() || 0}
                                </TableCell>
                              </TableRow>
                              <TableRow>
                                <TableCell sx={{ color: 'error.main', fontWeight: 'medium' }}>
                                  像素差異數量
                                </TableCell>
                                <TableCell align="right">
                                  {result.mask_statistics.pixel_diff_pixels?.toLocaleString() || 0}
                                </TableCell>
                              </TableRow>
                              <TableRow>
                                <TableCell sx={{ color: 'warning.main', fontWeight: 'medium' }}>
                                  圖像2獨有區域像素
                                </TableCell>
                                <TableCell align="right">
                                  {result.mask_statistics.diff_2_only_pixels?.toLocaleString() || 0}
                                </TableCell>
                              </TableRow>
                              <TableRow>
                                <TableCell sx={{ color: 'warning.main', fontWeight: 'medium' }}>
                                  圖像1獨有區域像素
                                </TableCell>
                                <TableCell align="right">
                                  {result.mask_statistics.diff_1_only_pixels?.toLocaleString() || 0}
                                </TableCell>
                              </TableRow>
                              <TableRow>
                                <TableCell sx={{ fontWeight: 'bold' }}>
                                  總印章像素
                                </TableCell>
                                <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                                  {result.mask_statistics.total_seal_pixels?.toLocaleString() || 0}
                                </TableCell>
                              </TableRow>

                              {/* 比例組 */}
                              <TableRow>
                                <TableCell colSpan={2} sx={{ backgroundColor: 'grey.100', fontWeight: 'bold', pt: 2 }}>
                                  比例
                                </TableCell>
                              </TableRow>
                              <TableRow>
                                <TableCell sx={{ color: 'success.main', fontWeight: 'medium' }}>
                                  重疊區域比例
                                </TableCell>
                                <TableCell align="right">
                                  {(result.mask_statistics.overlap_ratio * 100)?.toFixed(2) || '0.00'}%
                                </TableCell>
                              </TableRow>
                              <TableRow>
                                <TableCell sx={{ color: 'error.main', fontWeight: 'medium' }}>
                                  像素差異比例
                                </TableCell>
                                <TableCell align="right">
                                  {(result.mask_statistics.pixel_diff_ratio * 100)?.toFixed(2) || '0.00'}%
                                </TableCell>
                              </TableRow>
                              <TableRow>
                                <TableCell sx={{ color: 'warning.main', fontWeight: 'medium' }}>
                                  圖像2獨有區域比例
                                </TableCell>
                                <TableCell align="right">
                                  {(result.mask_statistics.diff_2_only_ratio * 100)?.toFixed(2) || '0.00'}%
                                </TableCell>
                              </TableRow>
                              <TableRow>
                                <TableCell sx={{ color: 'warning.main', fontWeight: 'medium' }}>
                                  圖像1獨有區域比例
                                </TableCell>
                                <TableCell align="right">
                                  {(result.mask_statistics.diff_1_only_ratio * 100)?.toFixed(2) || '0.00'}%
                                </TableCell>
                              </TableRow>
                              <TableRow>
                                <TableCell sx={{ color: 'error.main', fontWeight: 'bold' }}>
                                  總差異比例
                                </TableCell>
                                <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                                  {(result.mask_statistics.total_diff_ratio * 100)?.toFixed(2) || '0.00'}%
                                </TableCell>
                              </TableRow>
                            </TableBody>
                          </Table>
                        </TableContainer>
                      </AccordionDetails>
                    </Accordion>
                  </Box>
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

