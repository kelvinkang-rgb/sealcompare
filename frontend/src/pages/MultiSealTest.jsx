import React, { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
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
  DialogActions,
  DialogContentText,
  Snackbar,
  TextField,
  Slider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
  Collapse,
} from '@mui/material'
import { ExpandMore as ExpandMoreIcon, Settings as SettingsIcon, Error as ErrorIcon, ContentCopy as ContentCopyIcon, ExpandMore, ExpandLess, Info as InfoIcon } from '@mui/icons-material'
import { useMutation } from '@tanstack/react-query'
import { imageAPI } from '../services/api'
import ImagePreview from '../components/ImagePreview'
import SealDetectionBox from '../components/SealDetectionBox'
import MultiSealPreview from '../components/MultiSealPreview'
import MultiSealDetectionBox from '../components/MultiSealDetectionBox'
import MultiSealComparisonResults from '../components/MultiSealComparisonResults'
import SimilarityHistogram from '../components/SimilarityHistogram'

function MultiSealTest() {
  const navigate = useNavigate()
  
  // ==================== 圖像1相關狀態 ====================
  // 圖像1數據和檢測狀態
  const [image1, setImage1] = useState(null)
  const [showSealDialog1, setShowSealDialog1] = useState(false)
  const [sealDetectionResult1, setSealDetectionResult1] = useState(null)
  const [isDetecting1, setIsDetecting1] = useState(false)
  
  // ==================== 圖像2相關狀態 ====================
  // 圖像2數據和多印鑑檢測狀態
  const [image2, setImage2] = useState(null)
  const [showSealDialog2, setShowSealDialog2] = useState(false)
  const [multipleSeals, setMultipleSeals] = useState([])
  const [isDetecting2, setIsDetecting2] = useState(false)
  const image2InputRef = useRef(null) // 引用圖像2的 input 元素，用於重置文件選擇
  
  // ==================== 比對流程狀態 ====================
  // 裁切後的印鑑圖像ID列表
  const [croppedImageIds, setCroppedImageIds] = useState([])
  // 比對結果數據
  const [comparisonResults, setComparisonResults] = useState(null)
  // 比對任務狀態
  const [currentTaskUid, setCurrentTaskUid] = useState(null)
  const [taskStatus, setTaskStatus] = useState(null)
  
  // ==================== 比對參數設定 ====================
  // 相似度閾值（0-1，默認 0.88 即 88%）
  const [threshold, setThreshold] = useState(0.88)
  // 比對印鑑數量上限（默認 3）
  const [maxSeals, setMaxSeals] = useState(3)
  // 記錄上次檢測使用的 maxSeals，用於判斷是否需要重新檢測
  const [lastDetectionMaxSeals, setLastDetectionMaxSeals] = useState(null)
  // Mask相似度權重參數
  const [overlapWeight, setOverlapWeight] = useState(0.5) // 重疊區域權重，默認 50%
  const [pixelDiffPenaltyWeight, setPixelDiffPenaltyWeight] = useState(0.3) // 像素差異懲罰權重，默認 30%
  const [uniqueRegionPenaltyWeight, setUniqueRegionPenaltyWeight] = useState(0.2) // 獨有區域懲罰權重，默認 20%
  
  // ==================== UI狀態 ====================
  // 操作反饋提示
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' })
  // Dialog顯示狀態
  const [maskWeightDialogOpen, setMaskWeightDialogOpen] = useState(false)
  const [advancedSettingsOpen, setAdvancedSettingsOpen] = useState(false)
  // 錯誤詳情狀態
  const [lastError, setLastError] = useState(null)
  const [errorStage, setErrorStage] = useState(null) // 'save', 'crop', 'compare'
  const [errorDialogOpen, setErrorDialogOpen] = useState(false)
  const [errorDetailsExpanded, setErrorDetailsExpanded] = useState(false)

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
          setLastDetectionMaxSeals(maxSeals) // 記錄本次檢測使用的 maxSeals
          setShowSealDialog2(true)
        } else {
          setMultipleSeals([])
          setLastDetectionMaxSeals(maxSeals) // 記錄本次檢測使用的 maxSeals（即使未檢測到）
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

  // ==================== Mask相似度計算函數 ====================
  // 與後端 calculate_mask_based_similarity 邏輯一致
  const calculateMaskBasedSimilarity = (maskStats, overlapWeight, pixelDiffPenaltyWeight, uniqueRegionPenaltyWeight) => {
    if (!maskStats || maskStats.total_seal_pixels === 0) {
      return 0.0
    }
    
    try {
      // 1. 重疊區域獎勵
      const overlapRatio = maskStats.overlap_ratio || 0.0
      const overlapScore = overlapRatio
      
      // 2. 像素差異懲罰
      const pixelDiffRatio = maskStats.pixel_diff_ratio || 1.0
      const pixelDiffPenalty = 1.0 - pixelDiffRatio
      
      // 3. 獨有區域懲罰
      const diff1OnlyRatio = maskStats.diff_1_only_ratio || 0.0
      const diff2OnlyRatio = maskStats.diff_2_only_ratio || 0.0
      const uniquePenalty = 1.0 - (diff1OnlyRatio + diff2OnlyRatio)
      
      // 最終相似度計算
      const similarity = (
        overlapScore * overlapWeight +
        pixelDiffPenalty * pixelDiffPenaltyWeight +
        uniquePenalty * uniqueRegionPenaltyWeight
      )
      
      // 確保返回值在 [0.0, 1.0] 範圍內
      return Math.max(0.0, Math.min(1.0, similarity))
    } catch (error) {
      console.error('計算mask相似度失敗:', error)
      return 0.0
    }
  }

  // 輪詢任務結果（包括狀態和部分結果）- 合併為單一輪詢
  const { data: polledTaskResult } = useQuery({
    queryKey: ['task-result', currentTaskUid],
    queryFn: () => imageAPI.getTaskResult(currentTaskUid),
    enabled: !!currentTaskUid && taskStatus?.status !== 'completed' && taskStatus?.status !== 'failed',
    refetchInterval: (query) => {
      const data = query.state.data
      // 如果任務完成或失敗，停止輪詢
      if (data?.status === 'completed' || data?.status === 'failed') {
        return false
      }
      // 否則每 1.5 秒輪詢一次
      return 1500
    },
    refetchOnWindowFocus: true,
  })
  
  // 當輪詢結果更新時，同時更新狀態和結果顯示
  useEffect(() => {
    if (polledTaskResult) {
      // 更新任務狀態（從結果中提取狀態信息）
      const statusInfo = {
        status: polledTaskResult.status,
        total_count: polledTaskResult.total_count,
        success_count: polledTaskResult.success_count,
        error: polledTaskResult.error
      }
      setTaskStatus(statusInfo)
      
      // 處理任務失敗的情況
      if (polledTaskResult.status === 'failed') {
        setLastError({
          message: '比對任務失敗',
          detail: polledTaskResult.error || '未知錯誤',
          timestamp: new Date().toISOString(),
          taskUid: currentTaskUid
        })
        setErrorStage('compare')
        setSnackbar({
          open: true,
          message: `比對任務失敗 (UID: ${currentTaskUid?.substring(0, 8)}...)`,
          severity: 'error'
        })
      } else if (polledTaskResult.status === 'completed') {
        // 任務完成（不顯示 snackbar，因為 Alert 中已經顯示完成訊息）
        setLastError(null)
        setErrorStage(null)
      }
      
      // 處理結果顯示（即時顯示部分結果）
      if (polledTaskResult.results) {
        // 過濾出已完成疊圖的結果（有 overlay1_path 或 overlay2_path 的結果）
        // 這樣可以立即顯示已完成疊圖的結果，不需要等待所有比對完成
        const completedResults = polledTaskResult.results.filter(result => {
          // 只顯示已經完成疊圖的結果（有 overlay1_path 或 overlay2_path）
          // 或者有錯誤的結果（也需要顯示錯誤信息）
          return (result.overlay1_path || result.overlay2_path) || result.error
        })
        
        if (completedResults.length > 0) {
          // 對每個結果使用當前設定的參數動態計算 mask_based_similarity 和 is_match
          const processedResults = completedResults.map(result => {
            // 如果結果有錯誤，直接返回
            if (result.error) {
              return result
            }
            
            // 如果有 mask_statistics，使用當前設定的權重參數重新計算
            if (result.mask_statistics) {
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
                is_match: dynamicIsMatch
              }
            }
            
            // 如果沒有 mask_statistics，使用後端返回的值作為備用
            return result
          })
          
          // 確保結果按 seal_index 排序
          const sortedResults = [...processedResults].sort((a, b) => {
            const indexA = a.seal_index || 0
            const indexB = b.seal_index || 0
            return indexA - indexB
          })
          setComparisonResults(sortedResults)
        } else if (polledTaskResult.status === 'completed') {
          // 如果任務已完成但沒有結果，清空顯示（可能是所有結果都失敗了）
          setComparisonResults([])
        }
      }
    }
  }, [polledTaskResult, currentTaskUid, threshold, overlapWeight, pixelDiffPenaltyWeight, uniqueRegionPenaltyWeight])
  
  // 比對圖像1與多個印鑑（創建任務）
  const compareMutation = useMutation({
    mutationFn: ({ image1Id, sealImageIds, threshold, similaritySsimWeight, similarityTemplateWeight, pixelSimilarityWeight, histogramSimilarityWeight, overlapWeight, pixelDiffPenaltyWeight, uniqueRegionPenaltyWeight }) => 
      imageAPI.compareImage1WithSeals(image1Id, sealImageIds, threshold, similaritySsimWeight, similarityTemplateWeight, pixelSimilarityWeight, histogramSimilarityWeight, overlapWeight, pixelDiffPenaltyWeight, uniqueRegionPenaltyWeight),
    onSuccess: (taskData) => {
      // 保存任務 UID 並開始輪詢
      setCurrentTaskUid(taskData.task_uid)
      setTaskStatus({
        status: taskData.status
      })
      setLastError(null)
      setErrorStage(null)
      setSnackbar({
        open: true,
        message: `比對任務已創建 (UID: ${taskData.task_uid.substring(0, 8)}...)`,
        severity: 'info'
      })
    },
    onError: (error) => {
      // 構建詳細的錯誤信息
      const errorDetails = {
        message: error?.message || '未知錯誤',
        status: error?.response?.status,
        statusText: error?.response?.statusText,
        detail: error?.response?.data?.detail,
        data: error?.response?.data,
        config: {
          url: error?.config?.url,
          method: error?.config?.method,
          params: error?.config?.params,
          data: error?.config?.data
        },
        timestamp: new Date().toISOString(),
        sealCount: error?.config?.data?.seal_image_ids?.length || croppedImageIds.length,
        image1Id: error?.config?.data?.image1Id || uploadImage1Mutation.data?.id
      }
      
      // 記錄完整錯誤信息到 console
      console.error('比對失敗 - 完整錯誤信息:', errorDetails)
      console.error('比對失敗 - 錯誤對象:', error)
      console.error('比對失敗 - 錯誤堆疊:', error?.stack)
      
      // 保存錯誤詳情供顯示
      setLastError(errorDetails)
      setErrorStage('compare')
      
      // 顯示詳細錯誤訊息
      const errorMessage = errorDetails.detail || errorDetails.message || '比對失敗'
      setSnackbar({
        open: true,
        message: `比對失敗: ${errorMessage}`,
        severity: 'error'
      })
      
      // 清除比對結果（因為比對失敗）
      setComparisonResults(null)
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
      // 重置 input value，允許重新選擇相同文件
      if (image2InputRef.current) {
        image2InputRef.current.value = ''
      }
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
    setLastDetectionMaxSeals(null) // 重置追蹤狀態
    uploadImage2Mutation.reset()
    // 重置 input value，允許重新選擇文件
    if (image2InputRef.current) {
      image2InputRef.current.value = ''
    }
  }

  // 處理編輯圖像1印鑑
  const handleEditImage1Seal = () => {
    if (uploadImage1Mutation.data?.id) {
      setShowSealDialog1(true)
    }
  }

  // 處理編輯圖像2多印鑑
  const handleEditImage2Seals = async () => {
    if (uploadImage2Mutation.data?.id) {
      // 如果 maxSeals 改變了，重新觸發檢測
      if (lastDetectionMaxSeals !== null && lastDetectionMaxSeals !== maxSeals) {
        setIsDetecting2(true)
        try {
          const detectionResult = await imageAPI.detectMultipleSeals(
            uploadImage2Mutation.data.id, 
            maxSeals
          )
          if (detectionResult.detected && detectionResult.seals && detectionResult.seals.length > 0) {
            setMultipleSeals(detectionResult.seals)
            // 更新 image2 對象的 multiple_seals 屬性，確保 MultiSealPreview 能正確顯示
            setImage2(prev => prev ? { ...prev, multiple_seals: detectionResult.seals } : null)
            setLastDetectionMaxSeals(maxSeals)
            setSnackbar({
              open: true,
              message: `已使用新的上限值（${maxSeals}）重新檢測到 ${detectionResult.seals.length} 個印鑑`,
              severity: 'success'
            })
          } else {
            setMultipleSeals([])
            // 更新 image2 對象的 multiple_seals 屬性為空數組
            setImage2(prev => prev ? { ...prev, multiple_seals: [] } : null)
            setLastDetectionMaxSeals(maxSeals)
            setSnackbar({
              open: true,
              message: `已使用新的上限值（${maxSeals}）重新檢測，未檢測到印鑑`,
              severity: 'warning'
            })
          }
        } catch (error) {
          console.error('重新檢測失敗:', error)
          setSnackbar({
            open: true,
            message: '重新檢測失敗，請手動編輯',
            severity: 'error'
          })
        } finally {
          setIsDetecting2(false)
        }
      }
      setShowSealDialog2(true)
    }
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

    // 清除舊的比對結果和錯誤
    setComparisonResults(null)
    setLastError(null)
    setErrorStage(null)
    setCurrentTaskUid(null)
    setTaskStatus(null)

    try {
      // 2. 保存多印鑑位置（確保使用最新數據）
      setErrorStage('save')
      console.log('開始保存多印鑑位置...', { imageId: uploadImage2Mutation.data.id, sealCount: multipleSeals.length })
      await saveMultipleSealsMutation.mutateAsync({
        imageId: uploadImage2Mutation.data.id,
        seals: multipleSeals
      })
      console.log('保存多印鑑位置成功')

      // 3. 裁切印鑑（使用保存後的最新數據）
      setErrorStage('crop')
      console.log('開始裁切印鑑...', { imageId: uploadImage2Mutation.data.id, sealCount: multipleSeals.length })
      const cropResult = await cropSealsMutation.mutateAsync({
        imageId: uploadImage2Mutation.data.id,
        seals: image2?.multiple_seals || multipleSeals,
        margin: 10
      })
      console.log('裁切印鑑成功', { croppedCount: cropResult.cropped_image_ids?.length })

      // 4. 開始比對（使用最新的裁切圖像ID）
      if (cropResult.cropped_image_ids && cropResult.cropped_image_ids.length > 0) {
        setErrorStage('compare')
        console.log('開始比對...', { 
          image1Id: uploadImage1Mutation.data.id, 
          sealCount: cropResult.cropped_image_ids.length,
          threshold,
          maskWeights: {
            overlap: overlapWeight,
            pixelDiffPenalty: pixelDiffPenaltyWeight,
            uniqueRegionPenalty: uniqueRegionPenaltyWeight
          }
        })
        await compareMutation.mutateAsync({
          image1Id: uploadImage1Mutation.data.id,
          sealImageIds: cropResult.cropped_image_ids,
          threshold: threshold,
          similaritySsimWeight: 0.5, // 保留以向後兼容，但不再使用
          similarityTemplateWeight: 0.35,
          pixelSimilarityWeight: 0.1,
          histogramSimilarityWeight: 0.05,
          overlapWeight: overlapWeight,
          pixelDiffPenaltyWeight: pixelDiffPenaltyWeight,
          uniqueRegionPenaltyWeight: uniqueRegionPenaltyWeight
        })
        console.log('比對完成')
        setErrorStage(null) // 清除錯誤階段（成功）
      } else {
        const errorDetails = {
          message: '裁切失敗，無法進行比對',
          detail: '裁切操作未返回有效的圖像ID列表',
          timestamp: new Date().toISOString(),
          stage: 'crop'
        }
        setLastError(errorDetails)
        setErrorStage('crop')
        console.error('裁切失敗:', errorDetails)
        setSnackbar({
          open: true,
          message: '裁切失敗，無法進行比對',
          severity: 'error'
        })
      }
    } catch (error) {
      // 構建詳細的錯誤信息
      const errorDetails = {
        message: error?.message || '未知錯誤',
        status: error?.response?.status,
        statusText: error?.response?.statusText,
        detail: error?.response?.data?.detail,
        data: error?.response?.data,
        config: {
          url: error?.config?.url,
          method: error?.config?.method,
          params: error?.config?.params,
          data: error?.config?.data
        },
        timestamp: new Date().toISOString(),
        stage: errorStage || 'unknown'
      }
      
      // 記錄完整錯誤信息到 console
      console.error(`比對流程失敗 (階段: ${errorStage || 'unknown'}):`, errorDetails)
      console.error('比對流程失敗 - 錯誤對象:', error)
      console.error('比對流程失敗 - 錯誤堆疊:', error?.stack)
      
      // 保存錯誤詳情
      setLastError(errorDetails)
      
      // 根據階段顯示不同的錯誤訊息
      const stageMessages = {
        save: '保存印鑑位置失敗',
        crop: '裁切印鑑失敗',
        compare: '比對失敗',
        unknown: '比對流程失敗'
      }
      const stageMessage = stageMessages[errorStage] || stageMessages.unknown
      const errorMessage = errorDetails.detail || errorDetails.message || stageMessage
      
      setSnackbar({
        open: true,
        message: `${stageMessage}: ${errorMessage}`,
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
                max={160}
                step={1}
                marks={[
                  { value: 1, label: '1' },
                  { value: 3, label: '3' },
                  { value: 10, label: '10' },
                  { value: 50, label: '50' },
                  { value: 100, label: '100' },
                  { value: 160, label: '160' }
                ]}
                valueLabelDisplay="auto"
                sx={{ flex: 1 }}
              />
              <TextField
                type="number"
                value={maxSeals}
                onChange={(e) => {
                  const val = parseInt(e.target.value)
                  if (!isNaN(val) && val >= 1 && val <= 160) {
                    setMaxSeals(val)
                  }
                }}
                inputProps={{ 
                  min: 1, 
                  max: 160, 
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

          {/* Mask相似度權重參數設定 */}
          <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid', borderColor: 'divider' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Typography variant="h6">
                Mask相似度權重參數設定
              </Typography>
              <IconButton
                size="small"
                onClick={() => setMaskWeightDialogOpen(true)}
                sx={{ color: 'primary.main' }}
              >
                <InfoIcon />
              </IconButton>
            </Box>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
              設定Mask相似度計算時各項權重（總和建議為 1.0）
            </Typography>
            <Grid container spacing={2}>
          {/* 重疊區域權重 */}
          <Grid item xs={12} sm={6} md={4}>
            <Typography variant="body2" gutterBottom fontWeight="bold">
              重疊區域權重
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Slider
                value={overlapWeight}
                onChange={(e, value) => setOverlapWeight(value)}
                min={0}
                max={1}
                step={0.01}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${Math.round(value * 100)}%`}
                sx={{ flex: 1 }}
              />
              <TextField
                type="number"
                value={overlapWeight}
                onChange={(e) => {
                  const val = parseFloat(e.target.value)
                  if (!isNaN(val) && val >= 0 && val <= 1) {
                    setOverlapWeight(val)
                  }
                }}
                inputProps={{ min: 0, max: 1, step: 0.01 }}
                size="small"
                sx={{ width: '80px' }}
                label="權重"
              />
            </Box>
            <Typography variant="caption" color="text.secondary">
              當前: {Math.round(overlapWeight * 100)}%
            </Typography>
          </Grid>

          {/* 像素差異懲罰權重 */}
          <Grid item xs={12} sm={6} md={4}>
            <Typography variant="body2" gutterBottom fontWeight="bold">
              像素差異懲罰權重
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Slider
                value={pixelDiffPenaltyWeight}
                onChange={(e, value) => setPixelDiffPenaltyWeight(value)}
                min={0}
                max={1}
                step={0.01}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${Math.round(value * 100)}%`}
                sx={{ flex: 1 }}
              />
              <TextField
                type="number"
                value={pixelDiffPenaltyWeight}
                onChange={(e) => {
                  const val = parseFloat(e.target.value)
                  if (!isNaN(val) && val >= 0 && val <= 1) {
                    setPixelDiffPenaltyWeight(val)
                  }
                }}
                inputProps={{ min: 0, max: 1, step: 0.01 }}
                size="small"
                sx={{ width: '80px' }}
                label="權重"
              />
            </Box>
            <Typography variant="caption" color="text.secondary">
              當前: {Math.round(pixelDiffPenaltyWeight * 100)}%
            </Typography>
          </Grid>

          {/* 獨有區域懲罰權重 */}
          <Grid item xs={12} sm={6} md={4}>
            <Typography variant="body2" gutterBottom fontWeight="bold">
              獨有區域懲罰權重
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Slider
                value={uniqueRegionPenaltyWeight}
                onChange={(e, value) => setUniqueRegionPenaltyWeight(value)}
                min={0}
                max={1}
                step={0.01}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${Math.round(value * 100)}%`}
                sx={{ flex: 1 }}
              />
              <TextField
                type="number"
                value={uniqueRegionPenaltyWeight}
                onChange={(e) => {
                  const val = parseFloat(e.target.value)
                  if (!isNaN(val) && val >= 0 && val <= 1) {
                    setUniqueRegionPenaltyWeight(val)
                  }
                }}
                inputProps={{ min: 0, max: 1, step: 0.01 }}
                size="small"
                sx={{ width: '80px' }}
                label="權重"
              />
            </Box>
            <Typography variant="caption" color="text.secondary">
              當前: {Math.round(uniqueRegionPenaltyWeight * 100)}%
            </Typography>
          </Grid>
        </Grid>
        <Box sx={{ mt: 2 }}>
              <Typography variant="caption" color={Math.abs(overlapWeight + pixelDiffPenaltyWeight + uniqueRegionPenaltyWeight - 1.0) < 0.01 ? 'success.main' : 'warning.main'}>
                權重總和: {(overlapWeight + pixelDiffPenaltyWeight + uniqueRegionPenaltyWeight).toFixed(2)} 
                {Math.abs(overlapWeight + pixelDiffPenaltyWeight + uniqueRegionPenaltyWeight - 1.0) < 0.01 ? ' ✓' : ' (建議調整為 1.0)'}
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
                ref={image2InputRef}
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
              onSealClick={() => setShowSealDialog2(true)}
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
                      compareMutation.isPending ||
                      taskStatus?.status === 'processing'
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
                    {compareMutation.isPending && '正在創建比對任務...'}
                  </Alert>
                )}

                {/* 任務處理狀態 */}
                {currentTaskUid && taskStatus && (
                  <Alert 
                    severity={
                      taskStatus.status === 'completed' ? 'success' :
                      taskStatus.status === 'failed' ? 'error' :
                      taskStatus.status === 'processing' ? 'info' : 'warning'
                    } 
                    sx={{ mt: 1 }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {taskStatus.status === 'processing' && <CircularProgress size={16} />}
                      <Box>
                        <Typography variant="body2">
                          {taskStatus.status === 'pending' && '任務等待處理...'}
                          {taskStatus.status === 'processing' && `正在處理比對任務...`}
                          {taskStatus.status === 'completed' && `比對完成：${taskStatus.success_count}/${taskStatus.total_count} 個印鑑成功`}
                          {taskStatus.status === 'failed' && `比對失敗：${taskStatus.error || '未知錯誤'}`}
                        </Typography>
                        <Typography variant="caption" display="block" sx={{ mt: 0.5, fontFamily: 'monospace' }}>
                          任務 UID: {currentTaskUid}
                        </Typography>
                      </Box>
                    </Box>
                  </Alert>
                )}

                {/* 顯示裁切結果（僅在比對完成後顯示） */}
                {croppedImageIds.length > 0 && !compareMutation.isPending && taskStatus?.status !== 'processing' && (
                  <Alert severity="success" sx={{ mt: 2 }}>
                    已成功裁切 {croppedImageIds.length} 個印鑑圖像
                  </Alert>
                )}

                {/* 顯示錯誤詳情（當比對失敗時） */}
                {lastError && !compareMutation.isPending && taskStatus?.status !== 'processing' && (
                  <Alert 
                    severity="error" 
                    sx={{ mt: 2 }}
                    action={
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        {errorStage === 'compare' && croppedImageIds.length > 0 && (
                          <Button
                            color="inherit"
                            size="small"
                            onClick={() => {
                              // 重試比對（不需要重新裁切）
                              setLastError(null)
                              setErrorStage(null)
                              setCurrentTaskUid(null)
                              setTaskStatus(null)
                              compareMutation.mutate({
                                image1Id: uploadImage1Mutation.data.id,
                                sealImageIds: croppedImageIds,
                                threshold: threshold,
                                similaritySsimWeight: 0.5, // 保留以向後兼容，但不再使用
                                similarityTemplateWeight: 0.35,
                                pixelSimilarityWeight: 0.1,
                                histogramSimilarityWeight: 0.05,
                                overlapWeight: overlapWeight,
                                pixelDiffPenaltyWeight: pixelDiffPenaltyWeight,
                                uniqueRegionPenaltyWeight: uniqueRegionPenaltyWeight
                              })
                            }}
                            disabled={compareMutation.isPending}
                          >
                            重試比對
                          </Button>
                        )}
                        <Button
                          color="inherit"
                          size="small"
                          onClick={() => setErrorDialogOpen(true)}
                          startIcon={<ErrorIcon />}
                        >
                          查看詳情
                        </Button>
                      </Box>
                    }
                  >
                    {errorStage === 'save' && '保存印鑑位置失敗'}
                    {errorStage === 'crop' && '裁切印鑑失敗'}
                    {errorStage === 'compare' && '比對失敗'}
                    {!errorStage && '操作失敗'}
                    {lastError.detail && `: ${lastError.detail}`}
                    {lastError.taskUid && (
                      <Typography variant="caption" display="block" sx={{ mt: 0.5, fontFamily: 'monospace' }}>
                        任務 UID: {lastError.taskUid}
                      </Typography>
                    )}
                  </Alert>
                )}
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>

      {/* Histogram 統計圖表 */}
      {comparisonResults && comparisonResults.length > 0 && (
        <SimilarityHistogram
          results={comparisonResults}
        />
      )}

      {/* 顯示比對結果 */}
      {comparisonResults && comparisonResults.length > 0 && (
        <Box sx={{ mt: 4 }}>
          <MultiSealComparisonResults 
            results={comparisonResults}
            image1Id={uploadImage1Mutation.data?.id}
            similarityRange={null}
            threshold={threshold}
            overlapWeight={overlapWeight}
            pixelDiffPenaltyWeight={pixelDiffPenaltyWeight}
            uniqueRegionPenaltyWeight={uniqueRegionPenaltyWeight}
            calculateMaskBasedSimilarity={calculateMaskBasedSimilarity}
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

      {/* 錯誤詳情對話框 */}
      <Dialog
        open={errorDialogOpen}
        onClose={() => setErrorDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ErrorIcon color="error" />
            <Typography variant="h6">錯誤詳情</Typography>
          </Box>
        </DialogTitle>
        <DialogContent>
          {lastError && (
            <>
              <DialogContentText sx={{ mb: 2 }}>
                {errorStage === 'save' && '保存印鑑位置時發生錯誤'}
                {errorStage === 'crop' && '裁切印鑑時發生錯誤'}
                {errorStage === 'compare' && '比對過程中發生錯誤'}
                {!errorStage && '操作過程中發生錯誤'}
              </DialogContentText>

              <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  錯誤訊息
                </Typography>
                <Alert severity="error" sx={{ mb: 2 }}>
                  {lastError.detail || lastError.message || '未知錯誤'}
                </Alert>

                {lastError.status && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      HTTP 狀態
                    </Typography>
                    <Typography variant="body2">
                      {lastError.status} {lastError.statusText || ''}
                    </Typography>
                  </Box>
                )}

                {lastError.timestamp && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      發生時間
                    </Typography>
                    <Typography variant="body2">
                      {new Date(lastError.timestamp).toLocaleString('zh-TW')}
                    </Typography>
                  </Box>
                )}

                {lastError.sealCount && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      操作參數
                    </Typography>
                    <Typography variant="body2">
                      印鑑數量: {lastError.sealCount}
                      {lastError.image1Id && ` | 圖像1 ID: ${lastError.image1Id}`}
                    </Typography>
                  </Box>
                )}

                <Button
                  startIcon={errorDetailsExpanded ? <ExpandLess /> : <ExpandMore />}
                  onClick={() => setErrorDetailsExpanded(!errorDetailsExpanded)}
                  size="small"
                  sx={{ mb: 1 }}
                >
                  {errorDetailsExpanded ? '隱藏' : '顯示'}完整錯誤信息
                </Button>

                <Collapse in={errorDetailsExpanded}>
                  <Box sx={{ mt: 2, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      完整錯誤對象
                    </Typography>
                    <Box
                      component="pre"
                      sx={{
                        display: 'block',
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                        fontSize: '0.75rem',
                        maxHeight: '400px',
                        overflow: 'auto',
                        p: 1,
                        bgcolor: 'background.paper',
                        borderRadius: 1,
                        fontFamily: 'monospace',
                        border: '1px solid',
                        borderColor: 'divider'
                      }}
                    >
                      {JSON.stringify(lastError, null, 2)}
                    </Box>
                    <Button
                      startIcon={<ContentCopyIcon />}
                      onClick={() => {
                        navigator.clipboard.writeText(JSON.stringify(lastError, null, 2))
                        setSnackbar({
                          open: true,
                          message: '錯誤信息已複製到剪貼板',
                          severity: 'success'
                        })
                      }}
                      size="small"
                      sx={{ mt: 1 }}
                    >
                      複製錯誤信息
                    </Button>
                  </Box>
                </Collapse>

                <Box sx={{ mt: 3, p: 2, bgcolor: 'info.light', borderRadius: 1 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    如何查看後端日誌
                  </Typography>
                  <Typography variant="body2" component="div" sx={{ fontFamily: 'monospace', fontSize: '0.875rem' }}>
                    <Box component="div" sx={{ mb: 1 }}>
                      # 查看最近的日誌
                    </Box>
                    <Box component="div" sx={{ mb: 1 }}>
                      docker-compose logs --tail=100 backend
                    </Box>
                    <Box component="div" sx={{ mb: 1 }}>
                      # 實時查看日誌
                    </Box>
                    <Box component="div" sx={{ mb: 1 }}>
                      docker-compose logs -f backend
                    </Box>
                    <Box component="div" sx={{ mb: 1 }}>
                      # 查看錯誤日誌
                    </Box>
                    <Box component="div">
                      docker-compose logs --tail=100 backend | grep -i error
                    </Box>
                  </Typography>
                </Box>
              </Box>
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setErrorDialogOpen(false)}>關閉</Button>
        </DialogActions>
      </Dialog>

      {/* Mask相似度權重參數說明Dialog */}
      <Dialog open={maskWeightDialogOpen} onClose={() => setMaskWeightDialogOpen(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <InfoIcon color="primary" />
            Mask相似度權重參數說明
          </Box>
        </DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, mt: 1 }}>
            {/* 重疊區域權重說明 */}
            <Paper sx={{ p: 2, backgroundColor: '#e3f2fd' }}>
              <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
                重疊區域權重
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                <strong>預設值：</strong>0.5 (50%)
              </Typography>
              <Typography variant="body2" sx={{ mt: 1 }}>
                <strong>說明：</strong>
                <br />
                重疊區域比例越高，相似度越高。此權重控制重疊區域在最終相似度計算中的影響程度。
                <br />
                重疊區域是指兩個圖像中都有印鑑像素的區域，代表兩圖像的共同部分。
              </Typography>
            </Paper>

            {/* 像素差異懲罰權重說明 */}
            <Paper sx={{ p: 2, backgroundColor: '#fff3e0' }}>
              <Typography variant="h6" gutterBottom sx={{ color: 'warning.main' }}>
                像素差異懲罰權重
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                <strong>預設值：</strong>0.3 (30%)
              </Typography>
              <Typography variant="body2" sx={{ mt: 1 }}>
                <strong>說明：</strong>
                <br />
                重疊區域內的像素差異比例越高，相似度越低。此權重控制像素差異對相似度的懲罰程度。
                <br />
                即使兩個圖像有重疊區域，如果重疊區域內的像素值差異很大，也會降低相似度分數。
              </Typography>
            </Paper>

            {/* 獨有區域懲罰權重說明 */}
            <Paper sx={{ p: 2, backgroundColor: '#fce4ec' }}>
              <Typography variant="h6" gutterBottom sx={{ color: 'error.main' }}>
                獨有區域懲罰權重
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                <strong>預設值：</strong>0.2 (20%)
              </Typography>
              <Typography variant="body2" sx={{ mt: 1 }}>
                <strong>說明：</strong>
                <br />
                獨有區域比例越高，相似度越低。此權重控制獨有區域對相似度的懲罰程度。
                <br />
                獨有區域是指只在其中一個圖像中存在的印鑑像素區域，代表兩圖像的差異部分。
              </Typography>
            </Paper>

            {/* 計算公式說明 */}
            <Paper sx={{ p: 2, backgroundColor: '#f1f8e9' }}>
              <Typography variant="h6" gutterBottom sx={{ color: 'success.main' }}>
                計算公式
              </Typography>
              <Box
                sx={{
                  mt: 1.5,
                  p: 2,
                  backgroundColor: 'white',
                  borderRadius: 1,
                  border: '1px solid #e0e0e0',
                }}
              >
                <Typography
                  variant="body1"
                  sx={{ fontFamily: 'monospace', textAlign: 'center', fontWeight: 'bold' }}
                >
                  相似度 = 重疊區域分數 × 重疊區域權重 + 像素差異懲罰 × 像素差異懲罰權重 + 獨有區域懲罰 × 獨有區域懲罰權重
                </Typography>
              </Box>
              <Typography variant="body2" sx={{ mt: 1.5 }}>
                <strong>權重說明：</strong>
                <br />
                • <strong>重疊區域分數</strong>：重疊區域像素數 / 總印鑑像素數
                <br />
                • <strong>像素差異懲罰</strong>：1 - (重疊區域內像素差異比例)
                <br />
                • <strong>獨有區域懲罰</strong>：1 - (圖像1獨有區域比例 + 圖像2獨有區域比例)
              </Typography>
            </Paper>

            {/* 權重總和建議 */}
            <Paper sx={{ p: 2, backgroundColor: '#f3e5f5' }}>
              <Typography variant="h6" gutterBottom sx={{ color: 'secondary.main' }}>
                權重總和建議
              </Typography>
              <Typography variant="body2" sx={{ mt: 1 }}>
                三個權重參數的總和建議為 <strong>1.0</strong>，這樣可以確保各項指標在最終相似度計算中的比例平衡。
                <br />
                如果總和不為1.0，系統仍會正常運作，但可能會影響相似度分數的絕對值。
              </Typography>
            </Paper>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setMaskWeightDialogOpen(false)}>關閉</Button>
        </DialogActions>
      </Dialog>
    </Container>
  )
}

export default MultiSealTest

