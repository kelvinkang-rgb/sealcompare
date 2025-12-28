import React, { useState, useMemo } from 'react'
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
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Stack,
} from '@mui/material'
import { CheckCircle as CheckCircleIcon, Cancel as CancelIcon, ExpandMore as ExpandMoreIcon, Search as SearchIcon } from '@mui/icons-material'
import ImagePreviewDialog from './ImagePreviewDialog'

function MultiSealComparisonResults({ 
  results, 
  image1Id, 
  similarityRange,
  threshold = 0.83,
  overlapWeight = 0.5,
  pixelDiffPenaltyWeight = 0.3,
  uniqueRegionPenaltyWeight = 0.2,
  calculateMaskBasedSimilarity
}) {
  const [modalOpen, setModalOpen] = useState(false)
  const [modalImageUrl, setModalImageUrl] = useState('')
  const [modalImage, setModalImage] = useState(null)
  
  // 篩選、排序、搜尋狀態
  const [searchText, setSearchText] = useState('')
  const [matchFilter, setMatchFilter] = useState('all') // 'all', 'match', 'no-match'
  const [sortBy, setSortBy] = useState('index-asc') // 'index-asc', 'index-desc', 'similarity-asc', 'similarity-desc'
  const [minSimilarity, setMinSimilarity] = useState('') // mask相似度最小值（0-100）
  const [maxSimilarity, setMaxSimilarity] = useState('') // mask相似度最大值（0-100）
  
  // 動態計算每個結果的「主分數」與 is_match
  // - 優先使用後端回傳的 structure_similarity（對印泥深淺較不敏感，0-1）
  // - 若缺欄位（向後相容），才用 mask_based_similarity（可依權重重算）
  const processedResults = React.useMemo(() => {
    if (!results || results.length === 0) return []
    
    return results.map(result => {
      // 如果結果有錯誤，直接返回
      if (result.error) {
        return result
      }

      // 新主指標：structure_similarity
      if (result.structure_similarity !== null && result.structure_similarity !== undefined) {
        const primary = result.structure_similarity
        const dynamicIsMatch = primary >= threshold
        return {
          ...result,
          // 將 UI 既有使用的欄位覆寫成主分數，避免改動太大
          mask_based_similarity: primary,
          is_match: dynamicIsMatch,
          _primary_metric: 'structure_similarity'
        }
      }
      
      // 如果有 mask_statistics 和計算函數，使用當前設定的權重參數重新計算
      if (result.mask_statistics && calculateMaskBasedSimilarity) {
        const dynamicMaskSimilarity = calculateMaskBasedSimilarity(
          result.mask_statistics,
          overlapWeight,
          pixelDiffPenaltyWeight,
          uniqueRegionPenaltyWeight
        )
        
        // 使用當前設定的閾值重新判斷匹配狀態
        const dynamicIsMatch = dynamicMaskSimilarity >= threshold
        
        return {
          ...result,
          mask_based_similarity: dynamicMaskSimilarity,
          is_match: dynamicIsMatch,
          _primary_metric: 'mask_based_similarity'
        }
      }
      
      // 如果沒有 mask_statistics，使用結果中已有的值（向後兼容）
      const fallback = result.mask_based_similarity
      return {
        ...result,
        is_match: fallback !== null && fallback !== undefined ? fallback >= threshold : result.is_match,
        _primary_metric: 'mask_based_similarity'
      }
    })
  }, [results, threshold, overlapWeight, pixelDiffPenaltyWeight, uniqueRegionPenaltyWeight, calculateMaskBasedSimilarity])

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


  // 篩選、排序、搜尋邏輯
  const filteredAndSortedResults = useMemo(() => {
    if (!processedResults || processedResults.length === 0) return []

    let filtered = [...processedResults]

    // 1. 相似度範圍篩選（來自 histogram）
    if (similarityRange) {
      const [minSimilarity, maxSimilarity] = similarityRange
      filtered = filtered.filter(result => {
        if (result.mask_based_similarity === null || result.mask_based_similarity === undefined) return false
        return result.mask_based_similarity >= minSimilarity && result.mask_based_similarity <= maxSimilarity
      })
    }

    // 2. 匹配狀態篩選
    if (matchFilter !== 'all') {
      filtered = filtered.filter(result => {
        if (matchFilter === 'match') {
          return result.is_match === true
        } else if (matchFilter === 'no-match') {
          return result.is_match === false || result.error
        }
        return true
      })
    }

    // 3. Mask相似度範圍篩選
    if (minSimilarity !== '' || maxSimilarity !== '') {
      filtered = filtered.filter(result => {
        if (result.mask_based_similarity === null || result.mask_based_similarity === undefined) return false
        const similarityPercent = result.mask_based_similarity * 100
        const min = minSimilarity !== '' ? parseFloat(minSimilarity) : 0
        const max = maxSimilarity !== '' ? parseFloat(maxSimilarity) : 100
        return similarityPercent >= min && similarityPercent <= max
      })
    }

    // 4. 印鑑索引搜尋
    if (searchText.trim()) {
      const searchLower = searchText.toLowerCase().trim()
      filtered = filtered.filter(result => {
        return result.seal_index.toString().includes(searchLower)
      })
    }

    // 5. 排序
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'index-asc':
          return a.seal_index - b.seal_index
        case 'index-desc':
          return b.seal_index - a.seal_index
        case 'similarity-asc':
          const simA = a.mask_based_similarity ?? 0
          const simB = b.mask_based_similarity ?? 0
          return simA - simB
        case 'similarity-desc':
          const simA2 = a.mask_based_similarity ?? 0
          const simB2 = b.mask_based_similarity ?? 0
          return simB2 - simA2
        default:
          return 0
      }
    })

    return filtered
  }, [processedResults, similarityRange, matchFilter, searchText, sortBy, minSimilarity, maxSimilarity])

  if (!processedResults || processedResults.length === 0) {
    return (
      <Alert severity="info" sx={{ mt: 2 }}>
        暫無比對結果
      </Alert>
    )
  }

  return (
    <Box sx={{ mt: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">
          比對結果 ({filteredAndSortedResults.length} / {processedResults.length} 個印鑑)
        </Typography>
      </Box>

      {/* 篩選、排序、搜尋控制項 */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
          {/* 搜尋框 */}
          <TextField
            size="small"
            placeholder="搜尋印鑑索引..."
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            InputProps={{
              startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
            }}
            sx={{ flex: 1 }}
          />

          {/* Mask相似度範圍篩選 */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <TextField
              size="small"
              type="number"
              placeholder="最小%"
              value={minSimilarity}
              onChange={(e) => {
                const val = e.target.value
                if (val === '' || (parseFloat(val) >= 0 && parseFloat(val) <= 100)) {
                  setMinSimilarity(val)
                }
              }}
              inputProps={{ min: 0, max: 100, step: 0.1 }}
              sx={{ width: 80 }}
              label="最小相似度"
            />
            <Typography variant="body2" color="text.secondary">
              ~
            </Typography>
            <TextField
              size="small"
              type="number"
              placeholder="最大%"
              value={maxSimilarity}
              onChange={(e) => {
                const val = e.target.value
                if (val === '' || (parseFloat(val) >= 0 && parseFloat(val) <= 100)) {
                  setMaxSimilarity(val)
                }
              }}
              inputProps={{ min: 0, max: 100, step: 0.1 }}
              sx={{ width: 80 }}
              label="最大相似度"
            />
          </Box>

          {/* 匹配狀態篩選 */}
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel>匹配狀態</InputLabel>
            <Select
              value={matchFilter}
              label="匹配狀態"
              onChange={(e) => setMatchFilter(e.target.value)}
            >
              <MenuItem value="all">全部</MenuItem>
              <MenuItem value="match">匹配</MenuItem>
              <MenuItem value="no-match">不匹配</MenuItem>
            </Select>
          </FormControl>

          {/* 排序 */}
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel>排序方式</InputLabel>
            <Select
              value={sortBy}
              label="排序方式"
              onChange={(e) => setSortBy(e.target.value)}
            >
              <MenuItem value="index-asc">印鑑索引 ↑</MenuItem>
              <MenuItem value="index-desc">印鑑索引 ↓</MenuItem>
              <MenuItem value="similarity-asc">相似度 ↑</MenuItem>
              <MenuItem value="similarity-desc">相似度 ↓</MenuItem>
            </Select>
          </FormControl>
        </Stack>
      </Paper>
      
      <Grid container spacing={2}>
        {filteredAndSortedResults.map((result, index) => (
          <Grid item xs={12} key={result.seal_image_id || result.seal_index || index}>
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
                        label={`${result._primary_metric === 'structure_similarity' ? '結構相似度' : 'Mask相似度'}: ${(result.mask_based_similarity * 100).toFixed(2)}%`}
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
                          <Box sx={{ height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <img
                              src={getImageUrl(result.input_image1_path)}
                              alt="輸入圖像1"
                              loading="lazy"
                              decoding="async"
                              style={{
                                maxWidth: '100%',
                                maxHeight: '100%',
                                width: 'auto',
                                height: 'auto',
                                borderRadius: '4px',
                                objectFit: 'contain',
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
                          <Box sx={{ height: 180, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <img
                              src={getImageUrl(result.input_image2_path)}
                              alt="輸入圖像2"
                              loading="lazy"
                              decoding="async"
                              style={{
                                maxWidth: '100%',
                                maxHeight: '100%',
                                width: 'auto',
                                height: 'auto',
                                borderRadius: '4px',
                                objectFit: 'contain',
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
                            loading="lazy"
                            decoding="async"
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
                            loading="lazy"
                            decoding="async"
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
                    <Accordion defaultExpanded={false} TransitionProps={{ unmountOnExit: true }}>
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
                                    loading="lazy"
                                    decoding="async"
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
                                    loading="lazy"
                                    decoding="async"
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
                                    loading="lazy"
                                    decoding="async"
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
                                    loading="lazy"
                                    decoding="async"
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
                              loading="lazy"
                              decoding="async"
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

                {/* Mask統計資訊區域（不延後，立即可顯示） */}
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

                {/* 時間詳情區域 */}
                {!result.error && result.timing && (
                  <Box sx={{ mt: 2 }}>
                    <Accordion TransitionProps={{ unmountOnExit: true }}>
                      <AccordionSummary
                        expandIcon={<ExpandMoreIcon />}
                        aria-controls="timing-content"
                        id="timing-header"
                      >
                        <Typography variant="subtitle1" sx={{ fontWeight: 'medium' }}>
                          時間詳情
                        </Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <TableContainer component={Paper} variant="outlined">
                          <Table size="small">
                            <TableBody>
                              {(() => {
                                const timing = result.timing || {}
                                const alignmentStages = timing.alignment_stages || {}
                                const hasAlignmentStages = alignmentStages && typeof alignmentStages === 'object'

                                const formatSeconds = (v) => {
                                  const n = typeof v === 'number' ? v : 0
                                  return `${n.toFixed(2)} 秒`
                                }

                                const formatPercent = (part, total) => {
                                  const p = typeof part === 'number' ? part : 0
                                  const t = typeof total === 'number' ? total : 0
                                  if (!t || t <= 0) return ''
                                  return ` (${((p / t) * 100).toFixed(1)}%)`
                                }

                                // === 任務級主 key（維持現有顯示順序） ===
                                const displayedTimingKeys = new Set([
                                  'total',
                                  'load_images',
                                  'remove_bg_image1',
                                  'remove_bg_align_image2',
                                  'save_aligned_images',
                                  'similarity_calculation',
                                  'save_corrected_images',
                                  'create_overlay',
                                  'calculate_mask_stats',
                                  'create_heatmap',
                                  'alignment_stages',
                                ])

                                // === alignment_stages（依 seal_compare.py 結構化） ===
                                const bgStepDefs = [
                                  ['step1_convert_to_gray', '轉換灰度'],
                                  ['step2_detect_bg_color', '偵測背景色'],
                                  ['step3_otsu_threshold', 'OTSU二值化'],
                                  ['step4_combine_masks', '結合遮罩'],
                                  ['step5_morphology_bg', '形態學處理背景'],
                                  ['step6_contour_detection', '輪廓偵測'],
                                  ['step7_calculate_bbox', '計算邊界框'],
                                  ['step8_crop_image', '裁切圖像'],
                                  ['step9_remove_bg_final', '最終移除背景'],
                                ]

                                const stageDefs = [
                                  ['stage1_translation_coarse', '階段1：平移粗調'],
                                  ['stage2_rotation_coarse', '階段2：旋轉粗調'],
                                  ['stage3_translation_fine', '階段3：平移細調'],
                                  ['stage4_total', '階段4：旋轉細調與平移細調（總計）'],
                                  ['stage5_global_verification', '階段5：全局驗證'],
                                ]

                                const stage4SubDefs = [
                                  ['stage4_rotation_fine', '└─ 旋轉細調'],
                                  ['stage4_translation_fine', '└─ 平移細調'],
                                ]

                                const knownAlignmentKeys = new Set([
                                  'remove_background_total',
                                  'remove_background',
                                  ...bgStepDefs.map(([k]) => k),
                                  ...stageDefs.map(([k]) => k),
                                  ...stage4SubDefs.map(([k]) => k),
                                ])

                                const otherAlignmentEntries = hasAlignmentStages
                                  ? Object.entries(alignmentStages).filter(([k]) => !knownAlignmentKeys.has(k))
                                  : []

                                const otherTimingEntries = Object.entries(timing).filter(([k]) => !displayedTimingKeys.has(k))

                                const renderKeyValueRows = (entries, parentTotal) => {
                                  if (!entries || entries.length === 0) return null
                                  return entries.map(([k, v]) => (
                                    <TableRow key={k}>
                                      <TableCell sx={{ color: 'text.secondary' }}>{k}</TableCell>
                                      <TableCell align="right" sx={{ color: 'text.secondary' }}>
                                        {typeof v === 'number' ? formatSeconds(v) : JSON.stringify(v)}
                                        {typeof v === 'number' ? formatPercent(v, parentTotal) : ''}
                                      </TableCell>
                                    </TableRow>
                                  ))
                                }

                                return (
                                  <>
                                    {/* 總時間 */}
                                    {timing.total !== undefined && (
                                      <TableRow>
                                        <TableCell colSpan={2} sx={{ backgroundColor: 'primary.light', fontWeight: 'bold', color: 'primary.contrastText' }}>
                                          總時間: {typeof timing.total === 'number' ? timing.total.toFixed(2) : '0.00'} 秒
                                        </TableCell>
                                      </TableRow>
                                    )}

                                    {/* 任務級主階段 */}
                                    {timing.load_images !== undefined && (
                                      <TableRow>
                                        <TableCell>載入圖像</TableCell>
                                        <TableCell align="right">
                                          {formatSeconds(timing.load_images)}{formatPercent(timing.load_images, timing.total)}
                                        </TableCell>
                                      </TableRow>
                                    )}
                                    {timing.remove_bg_image1 !== undefined && (
                                      <TableRow>
                                        <TableCell>圖像1去背景</TableCell>
                                        <TableCell align="right">
                                          {formatSeconds(timing.remove_bg_image1)}{formatPercent(timing.remove_bg_image1, timing.total)}
                                        </TableCell>
                                      </TableRow>
                                    )}

                                    {/* 圖像2去背景 + 對齊（細節） */}
                                    {timing.remove_bg_align_image2 !== undefined && (
                                      <>
                                        <TableRow>
                                          <TableCell sx={{ fontWeight: 'medium' }}>圖像2去背景和對齊</TableCell>
                                          <TableCell align="right" sx={{ fontWeight: 'medium' }}>
                                            {formatSeconds(timing.remove_bg_align_image2)}{formatPercent(timing.remove_bg_align_image2, timing.total)}
                                          </TableCell>
                                        </TableRow>

                                        {hasAlignmentStages && (
                                          <>
                                            {/* 去背景（9步驟 + total） */}
                                            <TableRow>
                                              <TableCell sx={{ pl: 4, color: 'text.secondary', fontWeight: 'medium' }}>
                                                ├─ 去背景
                                              </TableCell>
                                              <TableCell align="right" sx={{ color: 'text.secondary', fontWeight: 'medium' }}>
                                                {formatSeconds(
                                                  alignmentStages.remove_background_total ??
                                                    alignmentStages.remove_background ??
                                                    0
                                                )}
                                                {formatPercent(
                                                  alignmentStages.remove_background_total ??
                                                    alignmentStages.remove_background ??
                                                    0,
                                                  timing.remove_bg_align_image2
                                                )}
                                              </TableCell>
                                            </TableRow>

                                            {bgStepDefs.map(([k, label], idx) => {
                                              const v = alignmentStages[k]
                                              const prefix = idx === bgStepDefs.length - 1 ? '│  └─' : '│  ├─'
                                              return (
                                                <TableRow key={k}>
                                                  <TableCell sx={{ pl: 6, color: 'text.secondary' }}>
                                                    {prefix} {label}
                                                  </TableCell>
                                                  <TableCell align="right" sx={{ color: 'text.secondary' }}>
                                                    {formatSeconds(v)}
                                                  </TableCell>
                                                </TableRow>
                                              )
                                            })}

                                            {/* 對齊階段 1-5 */}
                                            {stageDefs.map(([k, label]) => (
                                              <TableRow key={k}>
                                                <TableCell sx={{ pl: 4, color: 'text.secondary' }}>
                                                  ├─ {label}
                                                </TableCell>
                                                <TableCell align="right" sx={{ color: 'text.secondary' }}>
                                                  {formatSeconds(alignmentStages[k])}
                                                  {formatPercent(alignmentStages[k], timing.remove_bg_align_image2)}
                                                </TableCell>
                                              </TableRow>
                                            ))}

                                            {/* 階段4子步驟 */}
                                            {stage4SubDefs.map(([k, label]) => (
                                              <TableRow key={k}>
                                                <TableCell sx={{ pl: 6, color: 'text.secondary' }}>
                                                  │  {label}
                                                </TableCell>
                                                <TableCell align="right" sx={{ color: 'text.secondary' }}>
                                                  {formatSeconds(alignmentStages[k])}
                                                </TableCell>
                                              </TableRow>
                                            ))}

                                            {/* alignment_stages 其他未識別 key */}
                                            {otherAlignmentEntries.length > 0 && (
                                              <>
                                                <TableRow>
                                                  <TableCell colSpan={2} sx={{ pl: 4, backgroundColor: 'grey.100', fontWeight: 'bold' }}>
                                                    其他（alignment_stages）
                                                  </TableCell>
                                                </TableRow>
                                                {renderKeyValueRows(otherAlignmentEntries, timing.remove_bg_align_image2)}
                                              </>
                                            )}
                                          </>
                                        )}
                                      </>
                                    )}

                                    {timing.save_aligned_images !== undefined && (
                                      <TableRow>
                                        <TableCell>保存對齊圖像</TableCell>
                                        <TableCell align="right">
                                          {formatSeconds(timing.save_aligned_images)}{formatPercent(timing.save_aligned_images, timing.total)}
                                        </TableCell>
                                      </TableRow>
                                    )}
                                    {timing.similarity_calculation !== undefined && (
                                      <TableRow>
                                        <TableCell>相似度計算</TableCell>
                                        <TableCell align="right">
                                          {formatSeconds(timing.similarity_calculation)}{formatPercent(timing.similarity_calculation, timing.total)}
                                        </TableCell>
                                      </TableRow>
                                    )}
                                    {timing.save_corrected_images !== undefined && (
                                      <TableRow>
                                        <TableCell>保存校正圖像</TableCell>
                                        <TableCell align="right">
                                          {formatSeconds(timing.save_corrected_images)}{formatPercent(timing.save_corrected_images, timing.total)}
                                        </TableCell>
                                      </TableRow>
                                    )}
                                    {timing.create_overlay !== undefined && (
                                      <TableRow>
                                        <TableCell>生成疊圖</TableCell>
                                        <TableCell align="right">
                                          {formatSeconds(timing.create_overlay)}{formatPercent(timing.create_overlay, timing.total)}
                                        </TableCell>
                                      </TableRow>
                                    )}
                                    {timing.calculate_mask_stats !== undefined && (
                                      <TableRow>
                                        <TableCell>計算Mask統計</TableCell>
                                        <TableCell align="right">
                                          {formatSeconds(timing.calculate_mask_stats)}{formatPercent(timing.calculate_mask_stats, timing.total)}
                                        </TableCell>
                                      </TableRow>
                                    )}
                                    {timing.create_heatmap !== undefined && (
                                      <TableRow>
                                        <TableCell>生成熱力圖</TableCell>
                                        <TableCell align="right">
                                          {formatSeconds(timing.create_heatmap)}{formatPercent(timing.create_heatmap, timing.total)}
                                        </TableCell>
                                      </TableRow>
                                    )}

                                    {/* timing 其他未識別 key（全量呈現） */}
                                    {otherTimingEntries.length > 0 && (
                                      <>
                                        <TableRow>
                                          <TableCell colSpan={2} sx={{ backgroundColor: 'grey.100', fontWeight: 'bold' }}>
                                            其他（timing）
                                          </TableCell>
                                        </TableRow>
                                        {renderKeyValueRows(otherTimingEntries, timing.total)}
                                      </>
                                    )}
                                  </>
                                )
                              })()}
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

