import React, { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Container,
  Typography,
  Button,
  Box,
  Paper,
  Alert,
  CircularProgress,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  Snackbar,
  TextField,
  Slider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material'
import { ExpandMore as ExpandMoreIcon, Settings as SettingsIcon } from '@mui/icons-material'
import { useMutation } from '@tanstack/react-query'
import { imageAPI } from '../services/api'
import ImagePreview from '../components/ImagePreview'
import SealDetectionBox from '../components/SealDetectionBox'
import MultiSealPreview from '../components/MultiSealPreview'
import MultiSealDetectionBox from '../components/MultiSealDetectionBox'
import MultiSealComparisonResults from '../components/MultiSealComparisonResults'
import ImagePreviewDialog from '../components/ImagePreviewDialog'

function MultiSealTest() {
  const navigate = useNavigate()
  
  // 圖像1相關狀態
  const [image1, setImage1] = useState(null)
  const [showSealDialog1, setShowSealDialog1] = useState(false)
  const [sealDetectionResult1, setSealDetectionResult1] = useState(null)
  const [isDetecting1, setIsDetecting1] = useState(false)
  
  // 圖像2相關狀態
  const [image2, setImage2] = useState(null)
  const [showSealDialog2, setShowSealDialog2] = useState(false)
  const [multipleSeals, setMultipleSeals] = useState([])
  const [isDetecting2, setIsDetecting2] = useState(false)
  
  // 操作反饋
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' })
  const [croppedImageIds, setCroppedImageIds] = useState([])
  
  // 比對結果
  const [comparisonResults, setComparisonResults] = useState(null)
  
  // 相似度閾值
  const [threshold, setThreshold] = useState(0.5) // 默認 50%
  
  // 比對印鑑數量上限
  const [maxSeals, setMaxSeals] = useState(6) // 默認 6
  
  // 相似度權重參數
  const [similaritySsimWeight, setSimilaritySsimWeight] = useState(0.5) // 默認 50%
  const [similarityTemplateWeight, setSimilarityTemplateWeight] = useState(0.35) // 默認 35%
  const [pixelSimilarityWeight, setPixelSimilarityWeight] = useState(0.1) // 默認 10%
  const [histogramSimilarityWeight, setHistogramSimilarityWeight] = useState(0.05) // 默認 5%
  
  // 進階設定收折狀態
  const [advancedSettingsOpen, setAdvancedSettingsOpen] = useState(false)
  
  // 預覽對話框狀態
  const [previewImage1, setPreviewImage1] = useState(false)
  const [previewImage2, setPreviewImage2] = useState(false)

  // 上傳圖像1
  const uploadImage1Mutation = useMutation({
    mutationFn: imageAPI.upload,
    onSuccess: async (data) => {
      setImage1(data)
      // 自動檢測印鑑
      setIsDetecting1(true)
      try {
        const detectionResult = await imageAPI.detectSeal(data.id)
        setSealDetectionResult1(detectionResult)
        
        if (detectionResult.detected === true && detectionResult.bbox) {
          // 檢測成功，自動確認並保存結果
          try {
            const normalizedLocationData = {
              bbox: {
                x: Math.round(detectionResult.bbox.x),
                y: Math.round(detectionResult.bbox.y),
                width: Math.round(detectionResult.bbox.width),
                height: Math.round(detectionResult.bbox.height)
              },
              center: detectionResult.center ? {
                center_x: Math.round(detectionResult.center.center_x),
                center_y: Math.round(detectionResult.center.center_y),
                radius: typeof detectionResult.center.radius === 'number' ? detectionResult.center.radius : 0
              } : null,
              confidence: detectionResult.confidence || 1.0
            }
            
            await updateSealLocation1Mutation.mutateAsync({
              imageId: data.id,
              locationData: normalizedLocationData
            })
          } catch (error) {
            console.error('自動確認失敗:', error)
            setShowSealDialog1(true)
          }
        } else {
          setShowSealDialog1(true)
        }
      } catch (error) {
        console.error('檢測失敗:', error)
        setSealDetectionResult1({ detected: false, bbox: null, center: null })
        setShowSealDialog1(true)
      } finally {
        setIsDetecting1(false)
      }
    },
  })

  // 更新圖像1印鑑位置
  const updateSealLocation1Mutation = useMutation({
    mutationFn: ({ imageId, locationData }) => imageAPI.updateSealLocation(imageId, locationData),
    onSuccess: (data) => {
      setShowSealDialog1(false)
      setImage1(data)
    },
  })

  // 上傳圖像2
  const uploadImage2Mutation = useMutation({
    mutationFn: imageAPI.upload,
    onSuccess: async (data) => {
      setImage2(data)
      setCroppedImageIds([]) // 清除舊的裁切圖像ID
      // 自動檢測多個印鑑
      setIsDetecting2(true)
      try {
        const detectionResult = await imageAPI.detectMultipleSeals(data.id, maxSeals)
        if (detectionResult.detected && detectionResult.seals && detectionResult.seals.length > 0) {
          setMultipleSeals(detectionResult.seals)
          setShowSealDialog2(true)
        } else {
          setMultipleSeals([])
          setShowSealDialog2(true)
          setSnackbar({
            open: true,
            message: '未檢測到印鑑，請手動添加',
            severity: 'warning'
          })
        }
      } catch (error) {
        console.error('檢測失敗:', error)
        setMultipleSeals([])
        setShowSealDialog2(true)
        setSnackbar({
          open: true,
          message: '檢測失敗，請手動添加印鑑',
          severity: 'error'
        })
      } finally {
        setIsDetecting2(false)
      }
    },
  })

  // 保存多印鑑位置
  const saveMultipleSealsMutation = useMutation({
    mutationFn: ({ imageId, seals }) => imageAPI.saveMultipleSeals(imageId, seals),
    onSuccess: (data) => {
      setImage2(data)
      setCroppedImageIds([]) // 清除舊的裁切圖像ID，強制重新裁切
      // 同步更新 multipleSeals 狀態，確保與 image2.multiple_seals 一致
      if (data.multiple_seals && Array.isArray(data.multiple_seals)) {
        setMultipleSeals(data.multiple_seals)
      } else if (data.multiple_seals === null || data.multiple_seals === undefined) {
        setMultipleSeals([])
      }
      setShowSealDialog2(false)
      const savedCount = (data.multiple_seals && Array.isArray(data.multiple_seals)) 
        ? data.multiple_seals.length 
        : multipleSeals.length
      setSnackbar({
        open: true,
        message: `已保存 ${savedCount} 個印鑑位置`,
        severity: 'success'
      })
    },
    onError: (error) => {
      setSnackbar({
        open: true,
        message: error?.response?.data?.detail || '保存失敗',
        severity: 'error'
      })
    },
  })

  // 裁切印鑑
  const cropSealsMutation = useMutation({
    mutationFn: ({ imageId, seals, margin }) => imageAPI.cropSeals(imageId, seals, margin),
    onSuccess: (data) => {
      setCroppedImageIds(data.cropped_image_ids)
      setSnackbar({
        open: true,
        message: `成功裁切 ${data.count} 個印鑑圖像`,
        severity: 'success'
      })
    },
    onError: (error) => {
      setSnackbar({
        open: true,
        message: error?.response?.data?.detail || '裁切失敗',
        severity: 'error'
      })
    },
  })

  // 比對圖像1與多個印鑑
  const compareMutation = useMutation({
    mutationFn: ({ image1Id, sealImageIds, threshold, similaritySsimWeight, similarityTemplateWeight, pixelSimilarityWeight, histogramSimilarityWeight }) => 
      imageAPI.compareImage1WithSeals(image1Id, sealImageIds, threshold, similaritySsimWeight, similarityTemplateWeight, pixelSimilarityWeight, histogramSimilarityWeight),
    onSuccess: (data) => {
      setComparisonResults(data.results)
      setSnackbar({
        open: true,
        message: `比對完成：${data.success_count}/${data.total_count} 個印鑑成功`,
        severity: 'success'
      })
    },
    onError: (error) => {
      setSnackbar({
        open: true,
        message: error?.response?.data?.detail || '比對失敗',
        severity: 'error'
      })
    },
  })

  // 處理圖像1上傳
  const handleImage1Change = (e) => {
    const file = e.target.files[0]
    if (file) {
      uploadImage1Mutation.mutate(file)
    }
  }

  // 處理圖像2上傳
  const handleImage2Change = (e) => {
    const file = e.target.files[0]
    if (file) {
      uploadImage2Mutation.mutate(file)
    }
  }

  // 處理圖像1印鑑確認
  const handleSealConfirm1 = async (locationData) => {
    if (uploadImage1Mutation.data?.id) {
      try {
        const normalizedLocationData = {
          bbox: locationData.bbox ? {
            x: Math.round(locationData.bbox.x),
            y: Math.round(locationData.bbox.y),
            width: Math.round(locationData.bbox.width),
            height: Math.round(locationData.bbox.height)
          } : null,
          center: locationData.center ? {
            center_x: Math.round(locationData.center.center_x),
            center_y: Math.round(locationData.center.center_y),
            radius: typeof locationData.center.radius === 'number' ? locationData.center.radius : 0
          } : null,
          confidence: locationData.confidence || 1.0
        }
        
        await updateSealLocation1Mutation.mutateAsync({
          imageId: uploadImage1Mutation.data.id,
          locationData: normalizedLocationData
        })
      } catch (error) {
        console.error('更新印鑑位置失敗:', error)
        alert('更新印鑑位置失敗，請重試')
      }
    }
  }

  // 處理圖像2多印鑑確認
  const handleMultipleSealsConfirm = async (seals) => {
    if (uploadImage2Mutation.data?.id) {
      setMultipleSeals(seals)
      try {
        await saveMultipleSealsMutation.mutateAsync({
          imageId: uploadImage2Mutation.data.id,
          seals: seals
        })
      } catch (error) {
        console.error('保存多印鑑位置失敗:', error)
      }
    }
  }

  // 處理裁切
  const handleCropSeals = () => {
    if (!uploadImage2Mutation.data?.id || multipleSeals.length === 0) {
      setSnackbar({
        open: true,
        message: '請先完成多印鑑檢測',
        severity: 'warning'
      })
      return
    }

    cropSealsMutation.mutate({
      imageId: uploadImage2Mutation.data.id,
      seals: multipleSeals,
      margin: 10
    })
  }

  // 處理清除
  const handleClearImage1 = () => {
    setImage1(null)
    setSealDetectionResult1(null)
    setComparisonResults(null)
    uploadImage1Mutation.reset()
  }

  const handleClearImage2 = () => {
    setImage2(null)
    setMultipleSeals([])
    setCroppedImageIds([])
    setComparisonResults(null)
    uploadImage2Mutation.reset()
  }

  // 處理編輯圖像1印鑑
  const handleEditImage1Seal = () => {
    if (uploadImage1Mutation.data?.id) {
      setShowSealDialog1(true)
    }
  }

  // 處理編輯圖像2多印鑑
  const handleEditImage2Seals = () => {
    if (uploadImage2Mutation.data?.id) {
      setShowSealDialog2(true)
    }
  }

  // 處理點擊印鑑
  const handleSealClick = (index, seal) => {
    setShowSealDialog2(true)
  }

  // 處理預覽圖像1
  const handlePreviewImage1 = (image) => {
    if (image) {
      setPreviewImage1(true)
    }
  }

  // 處理預覽圖像2
  const handlePreviewImage2 = (image) => {
    if (image) {
      setPreviewImage2(true)
    }
  }

  // 處理開始比對
  const handleStartComparison = () => {
    if (!uploadImage1Mutation.data?.id) {
      setSnackbar({
        open: true,
        message: '請先上傳並標記圖像1',
        severity: 'warning'
      })
      return
    }

    if (!image1?.seal_bbox) {
      setSnackbar({
        open: true,
        message: '請先完成圖像1的印鑑位置標記',
        severity: 'warning'
      })
      return
    }

    if (croppedImageIds.length === 0) {
      setSnackbar({
        open: true,
        message: '請先完成印鑑裁切',
        severity: 'warning'
      })
      return
    }

    compareMutation.mutate({
      image1Id: uploadImage1Mutation.data.id,
      sealImageIds: croppedImageIds,
      threshold: threshold,
      similaritySsimWeight: similaritySsimWeight,
      similarityTemplateWeight: similarityTemplateWeight,
      pixelSimilarityWeight: pixelSimilarityWeight,
      histogramSimilarityWeight: histogramSimilarityWeight
    })
  }

  // 合併處理：保存、裁切、比對
  const handleCropAndCompare = async () => {
    // 1. 驗證前置條件
    if (!uploadImage1Mutation.data?.id || !image1?.seal_bbox) {
      setSnackbar({
        open: true,
        message: '請先上傳並標記圖像1的印鑑位置',
        severity: 'warning'
      })
      return
    }

    if (!uploadImage2Mutation.data?.id || multipleSeals.length === 0) {
      setSnackbar({
        open: true,
        message: '請先完成圖像2的多印鑑檢測',
        severity: 'warning'
      })
      return
    }

    // 清除舊的比對結果
    setComparisonResults(null)

    try {
      // 2. 保存多印鑑位置（確保使用最新數據）
      await saveMultipleSealsMutation.mutateAsync({
        imageId: uploadImage2Mutation.data.id,
        seals: multipleSeals
      })

      // 3. 裁切印鑑（使用保存後的最新數據）
      const cropResult = await cropSealsMutation.mutateAsync({
        imageId: uploadImage2Mutation.data.id,
        seals: image2?.multiple_seals || multipleSeals,
        margin: 10
      })

      // 4. 開始比對（使用最新的裁切圖像ID）
      if (cropResult.cropped_image_ids && cropResult.cropped_image_ids.length > 0) {
        await compareMutation.mutateAsync({
          image1Id: uploadImage1Mutation.data.id,
          sealImageIds: cropResult.cropped_image_ids,
          threshold: threshold,
          similaritySsimWeight: similaritySsimWeight,
          similarityTemplateWeight: similarityTemplateWeight,
          pixelSimilarityWeight: pixelSimilarityWeight,
          histogramSimilarityWeight: histogramSimilarityWeight
        })
      } else {
        setSnackbar({
          open: true,
          message: '裁切失敗，無法進行比對',
          severity: 'error'
        })
      }
    } catch (error) {
      console.error('比對流程失敗:', error)
      setSnackbar({
        open: true,
        message: error?.response?.data?.detail || '比對流程失敗',
        severity: 'error'
      })
    }
  }

  // 動態按鈕文字
  const getCompareButtonText = () => {
    if (saveMultipleSealsMutation.isPending) return '保存中...'
    if (cropSealsMutation.isPending) return '裁切中...'
    if (compareMutation.isPending) return '比對中...'
    return '開始比對多印鑑'
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Button onClick={() => navigate('/')} sx={{ mb: 2 }}>
          返回首頁
        </Button>
        <Typography variant="h4" component="h1" gutterBottom>
          多印鑑檢測測試
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          測試功能：圖像1使用單印鑑檢測，圖像2使用多印鑑檢測並裁切保存
        </Typography>
      </Box>

      {/* 進階設定 */}
      <Accordion 
        expanded={advancedSettingsOpen} 
        onChange={(e, expanded) => setAdvancedSettingsOpen(expanded)}
        sx={{ mb: 3 }}
      >
        <AccordionSummary
          expandIcon={<ExpandMoreIcon />}
          aria-controls="advanced-settings-content"
          id="advanced-settings-header"
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <SettingsIcon />
            <Typography variant="h6">
              進階設定
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
              （印鑑數量上限: {maxSeals} 個，相似度閾值: {Math.round(threshold * 100)}%）
            </Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          {/* 比對印鑑數量上限設定 */}
          <Box sx={{ mb: 3, p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid', borderColor: 'divider' }}>
            <Typography variant="body2" gutterBottom fontWeight="bold">
              比對印鑑數量上限
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
              設定圖像2中最多檢測的印鑑數量（當前: {maxSeals} 個）
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Slider
                value={maxSeals}
                onChange={(e, value) => setMaxSeals(value)}
                min={1}
                max={20}
                step={1}
                marks={[
                  { value: 1, label: '1' },
                  { value: 6, label: '6' },
                  { value: 10, label: '10' },
                  { value: 20, label: '20' }
                ]}
                valueLabelDisplay="auto"
                sx={{ flex: 1 }}
              />
              <TextField
                type="number"
                value={maxSeals}
                onChange={(e) => {
                  const val = parseInt(e.target.value)
                  if (!isNaN(val) && val >= 1 && val <= 20) {
                    setMaxSeals(val)
                  }
                }}
                inputProps={{ 
                  min: 1, 
                  max: 20, 
                  step: 1 
                }}
                size="small"
                sx={{ width: '100px' }}
                label="數量"
              />
            </Box>
          </Box>

          {/* 相似度閾值設定 */}
          <Box sx={{ mb: 3, p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid', borderColor: 'divider' }}>
            <Typography variant="body2" gutterBottom fontWeight="bold">
              相似度閾值設定
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
              設定比對時判斷為匹配的相似度閾值（當前: {Math.round(threshold * 100)}%）
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Slider
                value={threshold}
                onChange={(e, value) => setThreshold(value)}
                min={0}
                max={1}
                step={0.01}
                marks={[
                  { value: 0, label: '0%' },
                  { value: 0.5, label: '50%' },
                  { value: 1, label: '100%' }
                ]}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${Math.round(value * 100)}%`}
                sx={{ flex: 1 }}
              />
              <TextField
                type="number"
                value={threshold}
                onChange={(e) => {
                  const val = parseFloat(e.target.value)
                  if (!isNaN(val) && val >= 0 && val <= 1) {
                    setThreshold(val)
                  }
                }}
                inputProps={{ 
                  min: 0, 
                  max: 1, 
                  step: 0.01 
                }}
                size="small"
                sx={{ width: '100px' }}
                label="閾值"
              />
            </Box>
          </Box>

          {/* 相似度權重參數設定 */}
          <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid', borderColor: 'divider' }}>
            <Typography variant="h6" gutterBottom>
              相似度權重參數設定
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
              設定比對時各演算法的權重（總和建議為 1.0）
            </Typography>
            <Grid container spacing={2}>
          {/* SSIM 權重 */}
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="body2" gutterBottom fontWeight="bold">
              SSIM 權重
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Slider
                value={similaritySsimWeight}
                onChange={(e, value) => setSimilaritySsimWeight(value)}
                min={0}
                max={1}
                step={0.01}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${Math.round(value * 100)}%`}
                sx={{ flex: 1 }}
              />
              <TextField
                type="number"
                value={similaritySsimWeight}
                onChange={(e) => {
                  const val = parseFloat(e.target.value)
                  if (!isNaN(val) && val >= 0 && val <= 1) {
                    setSimilaritySsimWeight(val)
                  }
                }}
                inputProps={{ min: 0, max: 1, step: 0.01 }}
                size="small"
                sx={{ width: '80px' }}
                label="權重"
              />
            </Box>
            <Typography variant="caption" color="text.secondary">
              當前: {Math.round(similaritySsimWeight * 100)}%
            </Typography>
          </Grid>

          {/* Template Match 權重 */}
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="body2" gutterBottom fontWeight="bold">
              Template Match 權重
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Slider
                value={similarityTemplateWeight}
                onChange={(e, value) => setSimilarityTemplateWeight(value)}
                min={0}
                max={1}
                step={0.01}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${Math.round(value * 100)}%`}
                sx={{ flex: 1 }}
              />
              <TextField
                type="number"
                value={similarityTemplateWeight}
                onChange={(e) => {
                  const val = parseFloat(e.target.value)
                  if (!isNaN(val) && val >= 0 && val <= 1) {
                    setSimilarityTemplateWeight(val)
                  }
                }}
                inputProps={{ min: 0, max: 1, step: 0.01 }}
                size="small"
                sx={{ width: '80px' }}
                label="權重"
              />
            </Box>
            <Typography variant="caption" color="text.secondary">
              當前: {Math.round(similarityTemplateWeight * 100)}%
            </Typography>
          </Grid>

          {/* Pixel Similarity 權重 */}
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="body2" gutterBottom fontWeight="bold">
              Pixel Similarity 權重
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Slider
                value={pixelSimilarityWeight}
                onChange={(e, value) => setPixelSimilarityWeight(value)}
                min={0}
                max={1}
                step={0.01}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${Math.round(value * 100)}%`}
                sx={{ flex: 1 }}
              />
              <TextField
                type="number"
                value={pixelSimilarityWeight}
                onChange={(e) => {
                  const val = parseFloat(e.target.value)
                  if (!isNaN(val) && val >= 0 && val <= 1) {
                    setPixelSimilarityWeight(val)
                  }
                }}
                inputProps={{ min: 0, max: 1, step: 0.01 }}
                size="small"
                sx={{ width: '80px' }}
                label="權重"
              />
            </Box>
            <Typography variant="caption" color="text.secondary">
              當前: {Math.round(pixelSimilarityWeight * 100)}%
            </Typography>
          </Grid>

          {/* Histogram Similarity 權重 */}
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="body2" gutterBottom fontWeight="bold">
              Histogram Similarity 權重
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Slider
                value={histogramSimilarityWeight}
                onChange={(e, value) => setHistogramSimilarityWeight(value)}
                min={0}
                max={1}
                step={0.01}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${Math.round(value * 100)}%`}
                sx={{ flex: 1 }}
              />
              <TextField
                type="number"
                value={histogramSimilarityWeight}
                onChange={(e) => {
                  const val = parseFloat(e.target.value)
                  if (!isNaN(val) && val >= 0 && val <= 1) {
                    setHistogramSimilarityWeight(val)
                  }
                }}
                inputProps={{ min: 0, max: 1, step: 0.01 }}
                size="small"
                sx={{ width: '80px' }}
                label="權重"
              />
            </Box>
            <Typography variant="caption" color="text.secondary">
              當前: {Math.round(histogramSimilarityWeight * 100)}%
            </Typography>
          </Grid>
        </Grid>
        <Box sx={{ mt: 2 }}>
              <Typography variant="caption" color={Math.abs(similaritySsimWeight + similarityTemplateWeight + pixelSimilarityWeight + histogramSimilarityWeight - 1.0) < 0.01 ? 'success.main' : 'warning.main'}>
                權重總和: {(similaritySsimWeight + similarityTemplateWeight + pixelSimilarityWeight + histogramSimilarityWeight).toFixed(2)} 
                {Math.abs(similaritySsimWeight + similarityTemplateWeight + pixelSimilarityWeight + histogramSimilarityWeight - 1.0) < 0.01 ? ' ✓' : ' (建議調整為 1.0)'}
              </Typography>
            </Box>
          </Box>
        </AccordionDetails>
      </Accordion>

      <Grid container spacing={3}>
        {/* 圖像1區域 */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              圖像1（單印鑑）
            </Typography>
            <Box sx={{ mb: 2 }}>
              <input
                accept="image/*"
                style={{ display: 'none' }}
                id="image1-upload"
                type="file"
                onChange={handleImage1Change}
              />
              <label htmlFor="image1-upload">
                <Button
                  variant="outlined"
                  component="span"
                  fullWidth
                  disabled={uploadImage1Mutation.isPending || isDetecting1}
                  sx={{ mb: 1 }}
                  startIcon={
                    (uploadImage1Mutation.isPending || isDetecting1) && (
                      <CircularProgress size={16} />
                    )
                  }
                >
                  {uploadImage1Mutation.isPending
                    ? '上傳中...'
                    : isDetecting1
                    ? '檢測印鑑中...'
                    : '選擇圖像1'}
                </Button>
              </label>
              {(uploadImage1Mutation.isPending || isDetecting1) && (
                <Alert severity="info" sx={{ mt: 1 }}>
                  {uploadImage1Mutation.isPending ? '正在上傳圖像...' : '正在自動檢測印章位置...'}
                </Alert>
              )}
              {uploadImage1Mutation.isError && (
                <Alert severity="error" sx={{ mt: 1 }}>
                  上傳失敗
                </Alert>
              )}
              {uploadImage1Mutation.data && (
                <Button
                  variant="text"
                  size="small"
                  onClick={handleClearImage1}
                  sx={{ mt: 0.5 }}
                >
                  清除
                </Button>
              )}
            </Box>
            
            <ImagePreview
              image={image1}
              label="圖像1預覽"
              onEdit={handleEditImage1Seal}
              showSealIndicator={true}
              onPreview={handlePreviewImage1}
            />

            {/* 編輯印鑑位置按鈕 */}
            {image1 && (
              <Box sx={{ mt: 2 }}>
                <Button
                  variant="outlined"
                  onClick={handleEditImage1Seal}
                  fullWidth
                >
                  編輯印鑑位置
                </Button>
              </Box>
            )}
          </Paper>
        </Grid>

        {/* 圖像2區域 */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              圖像2（多印鑑）
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <input
                accept="image/*"
                style={{ display: 'none' }}
                id="image2-upload"
                type="file"
                onChange={handleImage2Change}
              />
              <label htmlFor="image2-upload">
                <Button
                  variant="outlined"
                  component="span"
                  fullWidth
                  disabled={uploadImage2Mutation.isPending || isDetecting2}
                  sx={{ mb: 1 }}
                  startIcon={
                    (uploadImage2Mutation.isPending || isDetecting2) && (
                      <CircularProgress size={16} />
                    )
                  }
                >
                  {uploadImage2Mutation.isPending
                    ? '上傳中...'
                    : isDetecting2
                    ? '檢測多印鑑中...'
                    : '選擇圖像2'}
                </Button>
              </label>
              {(uploadImage2Mutation.isPending || isDetecting2) && (
                <Alert severity="info" sx={{ mt: 1 }}>
                  {uploadImage2Mutation.isPending ? '正在上傳圖像...' : '正在自動檢測多個印章位置...'}
                </Alert>
              )}
              {uploadImage2Mutation.isError && (
                <Alert severity="error" sx={{ mt: 1 }}>
                  上傳失敗
                </Alert>
              )}
              {uploadImage2Mutation.data && (
                <Button
                  variant="text"
                  size="small"
                  onClick={handleClearImage2}
                  sx={{ mt: 0.5 }}
                >
                  清除
                </Button>
              )}
            </Box>
            
            <MultiSealPreview
              image={image2}
              seals={image2?.multiple_seals || multipleSeals || []}
              label="圖像2預覽"
              onSealClick={handleSealClick}
              onPreview={handlePreviewImage2}
            />

            {/* 操作按鈕 */}
            {multipleSeals.length > 0 && (
              <Box sx={{ mt: 2 }}>
                <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                  <Button
                    variant="outlined"
                    onClick={handleEditImage2Seals}
                    fullWidth
                  >
                    編輯印鑑位置
                  </Button>
                  <Button
                    variant="contained"
                    color="primary"
                    onClick={handleCropAndCompare}
                    disabled={
                      !uploadImage1Mutation.data?.id || 
                      !image1?.seal_bbox ||
                      saveMultipleSealsMutation.isPending ||
                      cropSealsMutation.isPending ||
                      compareMutation.isPending
                    }
                    fullWidth
                    startIcon={
                      (saveMultipleSealsMutation.isPending || 
                       cropSealsMutation.isPending || 
                       compareMutation.isPending) && (
                        <CircularProgress size={16} />
                      )
                    }
                  >
                    {getCompareButtonText()}
                  </Button>
                </Box>

                {/* 操作狀態提示 */}
                {(saveMultipleSealsMutation.isPending || 
                  cropSealsMutation.isPending || 
                  compareMutation.isPending) && (
                  <Alert severity="info" sx={{ mt: 1 }}>
                    {saveMultipleSealsMutation.isPending && '正在保存印鑑位置...'}
                    {cropSealsMutation.isPending && '正在裁切印鑑圖像...'}
                    {compareMutation.isPending && `正在比對圖像1與 ${croppedImageIds.length || multipleSeals.length} 個印鑑...`}
                  </Alert>
                )}

                {/* 顯示裁切結果（僅在比對完成後顯示） */}
                {croppedImageIds.length > 0 && !compareMutation.isPending && (
                  <Alert severity="success" sx={{ mt: 2 }}>
                    已成功裁切 {croppedImageIds.length} 個印鑑圖像
                  </Alert>
                )}
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* 顯示比對結果 */}
      {comparisonResults && comparisonResults.length > 0 && (
        <Box sx={{ mt: 4 }}>
          <MultiSealComparisonResults 
            results={comparisonResults}
            image1Id={uploadImage1Mutation.data?.id}
          />
        </Box>
      )}

      {/* 比對進行中提示（僅在比對階段顯示，保存和裁切階段已在按鈕區域顯示） */}
      {compareMutation.isPending && !saveMultipleSealsMutation.isPending && !cropSealsMutation.isPending && (
        <Box sx={{ mt: 4, textAlign: 'center' }}>
          <CircularProgress />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            正在比對圖像1與 {croppedImageIds.length || multipleSeals.length} 個印鑑，請稍候...
          </Typography>
        </Box>
      )}

      {/* 圖像1印鑑調整對話框 */}
      <Dialog
        open={showSealDialog1}
        onClose={() => {}}
        disableEscapeKeyDown={true}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>調整圖像1印鑑位置</DialogTitle>
        <DialogContent>
          {uploadImage1Mutation.data?.id && (
            <SealDetectionBox
              imageId={uploadImage1Mutation.data.id}
              initialBbox={image1?.seal_bbox || sealDetectionResult1?.bbox || uploadImage1Mutation.data?.seal_bbox || null}
              initialCenter={image1?.seal_center || sealDetectionResult1?.center || uploadImage1Mutation.data?.seal_center || null}
              onConfirm={handleSealConfirm1}
              onCancel={() => setShowSealDialog1(false)}
            />
          )}
        </DialogContent>
      </Dialog>

      {/* 圖像2多印鑑調整對話框 */}
      <Dialog
        open={showSealDialog2}
        onClose={() => {}}
        disableEscapeKeyDown={true}
        maxWidth="lg"
        fullWidth
      >
        <DialogTitle>調整圖像2多印鑑位置</DialogTitle>
        <DialogContent>
          {uploadImage2Mutation.data?.id && (
            <MultiSealDetectionBox
              imageId={uploadImage2Mutation.data.id}
              initialSeals={image2?.multiple_seals || multipleSeals || []}
              onConfirm={handleMultipleSealsConfirm}
              onCancel={() => setShowSealDialog2(false)}
            />
          )}
        </DialogContent>
      </Dialog>

      {/* 圖像1預覽對話框 */}
      <ImagePreviewDialog
        open={previewImage1}
        onClose={() => setPreviewImage1(false)}
        image={image1}
        sealBbox={image1?.seal_bbox || null}
        seals={[]}
      />

      {/* 圖像2預覽對話框 */}
      <ImagePreviewDialog
        open={previewImage2}
        onClose={() => setPreviewImage2(false)}
        image={image2}
        sealBbox={null}
        seals={image2?.multiple_seals || multipleSeals || []}
      />

      {/* 操作反饋提示 */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setSnackbar({ ...snackbar, open: false })}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Container>
  )
}

export default MultiSealTest

