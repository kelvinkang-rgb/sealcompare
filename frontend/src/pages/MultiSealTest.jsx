import React, { useState, useRef, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Container,
  Typography,
  Button,
  Box,
  Paper,
  Alert,
  CircularProgress,
  LinearProgress,
  Grid,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  DialogContentText,
  Snackbar,
  TextField,
  MenuItem,
  Stack,
  FormControl,
  InputLabel,
  Select,
  Slider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  IconButton,
  Collapse,
} from '@mui/material'
import { ExpandMore as ExpandMoreIcon, Settings as SettingsIcon, Error as ErrorIcon, ContentCopy as ContentCopyIcon, ExpandMore, ExpandLess, Info as InfoIcon, Search as SearchIcon } from '@mui/icons-material'
import { useMutation } from '@tanstack/react-query'
import { imageAPI } from '../services/api'
import ImagePreview from '../components/ImagePreview'
import SealDetectionBox from '../components/SealDetectionBox'
import MultiSealPreview from '../components/MultiSealPreview'
import MultiSealDetectionBox from '../components/MultiSealDetectionBox'
import PdfPagePicker from '../components/PdfPagePicker'
import MultiSealComparisonResults from '../components/MultiSealComparisonResults'
import SimilarityHistogram from '../components/SimilarityHistogram'
import { useFeatureFlag, FEATURE_FLAGS } from '../config/featureFlags'

function MultiSealTest() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  
  // 功能開關
  const showSimilarityHistogram = useFeatureFlag(FEATURE_FLAGS.SIMILARITY_HISTOGRAM)
  const showAdvancedSettings = useFeatureFlag(FEATURE_FLAGS.ADVANCED_SETTINGS)
  const showMaxSealsSetting = useFeatureFlag(FEATURE_FLAGS.MAX_SEALS_SETTING)
  const showThresholdSetting = useFeatureFlag(FEATURE_FLAGS.THRESHOLD_SETTING)
  const showMaskWeightsSetting = useFeatureFlag(FEATURE_FLAGS.MASK_WEIGHTS_SETTING)
  const showTaskTimingStatistics = useFeatureFlag(FEATURE_FLAGS.TASK_TIMING_STATISTICS)
  
  // ==================== 圖像1相關狀態 ====================
  // 圖像1數據和檢測狀態
  const [image1, setImage1] = useState(null)
  // 圖像1為 PDF 時：選定的模板頁（page image）
  const [image1TemplatePage, setImage1TemplatePage] = useState(null)
  const [image1TemplatePageId, setImage1TemplatePageId] = useState(null)
  const [image1PreferredPageId, setImage1PreferredPageId] = useState(null)
  const [showSealDialog1, setShowSealDialog1] = useState(false)
  const [sealDetectionResult1, setSealDetectionResult1] = useState(null)
  const [isDetecting1, setIsDetecting1] = useState(false)
  
  // ==================== 圖像2相關狀態 ====================
  // 圖像2數據和多印鑑檢測狀態
  const [image2, setImage2] = useState(null)
  // 圖像2為 PDF 時：選定的預覽/編輯頁（page image）
  const [image2SelectedPage, setImage2SelectedPage] = useState(null)
  const [image2SelectedPageId, setImage2SelectedPageId] = useState(null)
  const [image2PreferredPageId, setImage2PreferredPageId] = useState(null)
  const [showSealDialog2, setShowSealDialog2] = useState(false)
  const [multipleSeals, setMultipleSeals] = useState([])
  const [pdfPageDetections2, setPdfPageDetections2] = useState(null) // PDF 圖像2：每頁偵測結果
  const [isDetecting2, setIsDetecting2] = useState(false)
  const image2InputRef = useRef(null) // 引用圖像2的 input 元素，用於重置文件選擇
  
  // ==================== 比對流程狀態 ====================
  // 裁切後的印鑑圖像ID列表
  const [croppedImageIds, setCroppedImageIds] = useState([])
  // 比對結果數據
  const [comparisonResults, setComparisonResults] = useState(null)
  const lastCompletedCountRef = useRef(0)
  // 比對任務狀態
  const [currentTaskUid, setCurrentTaskUid] = useState(null)
  const [taskStatus, setTaskStatus] = useState(null)
  // PDF 比對任務狀態（image1 第1頁 vs image2 全部頁）
  const [pdfTaskUid, setPdfTaskUid] = useState(null)
  const [pdfTaskStatus, setPdfTaskStatus] = useState(null)
  // PDF 比對結果數據（處理後的結果，對齊 PNG/JPG 的 comparisonResults 更新策略）
  const [pdfComparisonResults, setPdfComparisonResults] = useState(null)
  // PDF 全頁共用篩選器（供每頁結果共用）
  const [pdfGlobalFilter, setPdfGlobalFilter] = useState({
    searchText: '',
    matchFilter: 'all', // 'all' | 'match' | 'no-match'
    sortBy: 'index-asc', // 'index-asc' | 'index-desc' | 'similarity-asc' | 'similarity-desc'
    minSimilarity: '', // 0-100 (%)
    maxSimilarity: '', // 0-100 (%)
    similarityRange: null, // [min,max] 0-1（由 histogram 點選回寫）
  })

  const resetPdfTaskUI = useCallback(() => {
    setPdfTaskUid(null)
    setPdfTaskStatus(null)
    setPdfComparisonResults(null)
    setPdfGlobalFilter({
      searchText: '',
      matchFilter: 'all',
      sortBy: 'index-asc',
      minSimilarity: '',
      maxSimilarity: '',
      similarityRange: null,
    })
  }, [])

  const clamp01 = useCallback((v) => {
    if (v === null || v === undefined) return undefined
    const n = Number(v)
    if (!Number.isFinite(n)) return undefined
    return Math.max(0, Math.min(1, n))
  }, [])
  
  // ==================== 比對參數設定 ====================
  // 相似度閾值（0-1，默認 0.83 即 83%）
  const [threshold, setThreshold] = useState(0.83)
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

  const image1Effective = image1?.is_pdf ? image1TemplatePage : image1
  const image1EffectiveId = image1?.is_pdf ? image1TemplatePageId : image1?.id
  const image1HasSeal = !!image1Effective?.seal_bbox

  const image2Effective = image2?.is_pdf ? image2SelectedPage : image2
  const image2EffectiveId = image2?.is_pdf ? image2SelectedPageId : image2?.id
  const image2PageDetection = Array.isArray(pdfPageDetections2)
    ? pdfPageDetections2.find(p => String(p.page_image_id) === String(image2SelectedPageId))
    : null
  const image2EffectiveSeals = image2?.is_pdf
    ? (image2SelectedPage?.multiple_seals || image2PageDetection?.seals || [])
    : (image2?.multiple_seals || multipleSeals || [])

  // 上傳圖像1
  const uploadImage1Mutation = useMutation({
    mutationFn: imageAPI.upload,
    onSuccess: async (data) => {
      setImage1(data)
      setImage1TemplatePage(null)
      setImage1TemplatePageId(null)
      setImage1PreferredPageId(null)
      // PDF：逐頁偵測並跳到建議頁，再讓使用者手動框選微調
      if (data?.is_pdf) {
        setIsDetecting1(true)
        try {
          const detectionResult = await imageAPI.detectSeal(data.id)
          setSealDetectionResult1(detectionResult)
          const pageId = detectionResult?.page_image_id || data?.pages?.[0]?.id || null
          setImage1PreferredPageId(pageId)
          if (pageId) {
            const pageImage = await imageAPI.get(pageId)
            setImage1TemplatePage(pageImage)
            setImage1TemplatePageId(pageId)
            setShowSealDialog1(true)
            setSnackbar({
              open: true,
              message: `圖像1 PDF 已自動跳到第 ${detectionResult?.page_index || pageImage?.page_index || '?'} 頁，請確認/微調模板印鑑框`,
              severity: 'info'
            })
          } else {
            setSnackbar({
              open: true,
              message: `圖像1 已上傳 PDF（共 ${data.pdf_page_count || data.pages?.length || 0} 頁），但未取得分頁，無法預覽/框選`,
              severity: 'warning'
            })
          }
        } catch (error) {
          console.error('PDF 偵測失敗:', error)
          setSealDetectionResult1({ detected: false, bbox: null, center: null })
          setSnackbar({
            open: true,
            message: 'PDF 偵測失敗，請改用手動選頁後框選（或檢查後端日誌）',
            severity: 'error'
          })
        } finally {
          setIsDetecting1(false)
        }
        return
      }
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
              confidence: clamp01(detectionResult.confidence) ?? 1.0
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
      if (image1?.is_pdf) {
        setImage1TemplatePage(data)
        setImage1TemplatePageId(data?.id || image1TemplatePageId)
      } else {
        setImage1(data)
      }
    },
  })

  // 上傳圖像2
  const uploadImage2Mutation = useMutation({
    mutationFn: imageAPI.upload,
    onSuccess: async (data) => {
      setImage2(data)
      setImage2SelectedPage(null)
      setImage2SelectedPageId(null)
      setImage2PreferredPageId(null)
      setCroppedImageIds([]) // 清除舊的裁切圖像ID
      setPdfPageDetections2(null)
      // 自動檢測多個印鑑
      setIsDetecting2(true)
      try {
        const detectionResult = await imageAPI.detectMultipleSeals(data.id, maxSeals)
        // PDF：後端會回傳 pages
        if (data?.is_pdf && Array.isArray(detectionResult.pages)) {
          setPdfPageDetections2(detectionResult.pages)
          setMultipleSeals([])
          // 預設跳到第一個有偵測到印鑑的頁，否則第 1 頁
          const preferred = detectionResult.pages.find(p => p.detected && (p.count > 0 || (p.seals && p.seals.length > 0)))
          const preferredPageId = preferred?.page_image_id || detectionResult.pages[0]?.page_image_id || data?.pages?.[0]?.id || null
          setImage2PreferredPageId(preferredPageId)
          setImage2SelectedPageId(preferredPageId)
          if (preferredPageId) {
            try {
              const pageImage = await imageAPI.get(preferredPageId)
              setImage2SelectedPage(pageImage)
            } catch (e) {
              // 即使拉取頁面失敗，也保留摘要
              console.error('拉取 PDF 頁面失敗:', e)
            }
          }
          setLastDetectionMaxSeals(maxSeals) // 記錄本次檢測使用的 maxSeals
          setSnackbar({
            open: true,
            message: `PDF 圖像2 逐頁偵測完成（共 ${detectionResult.pages.length} 頁，總計 ${detectionResult.total_count ?? detectionResult.count ?? 0} 個印鑑）`,
            severity: 'success'
          })
          return
        }

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
  // 允許已完成任務至少查詢一次結果（用於顯示最終結果）
  const { data: polledTaskResult } = useQuery({
    queryKey: ['task-result', currentTaskUid],
    queryFn: () => imageAPI.getTaskResult(currentTaskUid),
    enabled: !!currentTaskUid,
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

  // ===== PDF 任務輪詢（image1 第1頁 vs image2 全部頁）=====
  const { data: polledPdfTaskStatus } = useQuery({
    queryKey: ['pdf-task-status', pdfTaskUid],
    queryFn: () => imageAPI.getPdfTaskStatus(pdfTaskUid),
    enabled: !!pdfTaskUid,
    refetchInterval: (query) => {
      const s = query.state.data?.status
      if (s === 'completed' || s === 'failed') {
        return false
      }
      return 1500
    },
    // E2E/headless 或背景分頁時避免 interval 被暫停，確保 UI 能同步到最終狀態
    refetchIntervalInBackground: true,
    refetchOnWindowFocus: true,
  })

  // PDF 任務結果查詢：允許已完成任務至少查詢一次結果（用於顯示最終結果）
  // 任務進行中時定期查詢以獲取增量結果（後端有增量寫入）
  const { data: polledPdfTaskResult } = useQuery({
    queryKey: ['pdf-task-result', pdfTaskUid],
    queryFn: () => imageAPI.getPdfTaskResult(pdfTaskUid),
    enabled: !!pdfTaskUid,
    refetchOnMount: 'always',
    refetchInterval: (query) => {
      const resultData = query.state.data
      const status = polledPdfTaskStatus?.status || resultData?.status
      // 如果任務完成或失敗，停止輪詢
      if (status === 'completed' || status === 'failed') {
        return false
      }
      // pending/processing：定期查詢結果（後端有增量寫入，可以即時顯示部分結果）
      if (status === 'pending' || status === 'processing') {
        return 2000 // 每 2 秒查詢一次，獲取增量結果
      }
      // 其他情況（pending）不輪詢
      return false
    },
    // 同上：避免 background 時 interval 暫停
    refetchIntervalInBackground: true,
    refetchOnWindowFocus: true,
  })

  useEffect(() => {
    if (polledPdfTaskStatus) {
      setPdfTaskStatus(polledPdfTaskStatus)
      // 狀態變化時同步觸發結果查詢：
      // - pending -> processing：開始拿增量結果
      // - completed/failed：確保拿到最終結果
      if (pdfTaskUid) {
      if (polledPdfTaskStatus.status === 'completed' || polledPdfTaskStatus.status === 'failed') {
          queryClient.refetchQueries({ queryKey: ['pdf-task-result', pdfTaskUid] })
        } else {
        queryClient.invalidateQueries({ queryKey: ['pdf-task-result', pdfTaskUid] })
        }
      }
    }
  }, [polledPdfTaskStatus, pdfTaskUid, queryClient])

  // 處理 PDF 比對結果（對齊 PNG/JPG 的處理邏輯：只顯示已生成疊圖或有錯誤的結果，增量合併，completed 強制更新）
  useEffect(() => {
    if (!polledPdfTaskResult) return

    // 1) 同步狀態（從結果中提取狀態信息，避免 status polling 停在舊狀態）
    if (polledPdfTaskResult.status) {
      const pagesDone = Number(polledPdfTaskResult.pages_done)
      const pagesTotal = Number(polledPdfTaskResult.pages_total)
      const progressFromCounts =
        Number.isFinite(pagesDone) && Number.isFinite(pagesTotal) && pagesTotal > 0
          ? (pagesDone / pagesTotal) * 100
          : undefined

      const isTerminal = (s) => s === 'completed' || s === 'failed'
      setPdfTaskStatus((prev) => {
        const prevStatus = prev?.status
        const nextStatus = polledPdfTaskResult.status
        return {
        ...(prev || {}),
          // 避免「已 completed」被結果輪詢的舊 processing 回寫降級
          status: isTerminal(prevStatus) ? prevStatus : nextStatus,
        progress: progressFromCounts ?? prev?.progress,
        }
      })
    }

    // 2) 結果處理：只顯示已生成疊圖（overlay1/overlay2）或 error 的項目
    const rawPages = Array.isArray(polledPdfTaskResult.results_by_page)
      ? polledPdfTaskResult.results_by_page
      : []

    const normalizeResults = (page) =>
      Array.isArray(page?.results)
        ? page.results
        : (page?.results && Array.isArray(page.results.results) ? page.results.results : [])

    const processedPages = rawPages.map((page) => {
      const pageResults = normalizeResults(page)
      const completedResults = pageResults.filter((r) => (r?.overlay1_path || r?.overlay2_path) || r?.error)

      const processedResults = completedResults.map((r) => {
        if (r?.error) return r

        // 新主指標：structure_similarity
        if (r.structure_similarity !== null && r.structure_similarity !== undefined) {
          const primary = r.structure_similarity
          const dynamicIsMatch = primary >= threshold
          return {
            ...r,
            mask_based_similarity: primary,
            is_match: dynamicIsMatch,
            _primary_metric: 'structure_similarity',
          }
        }

        if (r.mask_statistics) {
          const dynamicMaskSimilarity = calculateMaskBasedSimilarity(
            r.mask_statistics,
            overlapWeight,
            pixelDiffPenaltyWeight,
            uniqueRegionPenaltyWeight
          )
          const dynamicIsMatch = dynamicMaskSimilarity >= threshold
          return {
            ...r,
            mask_based_similarity: dynamicMaskSimilarity,
            is_match: dynamicIsMatch,
            _primary_metric: 'mask_based_similarity',
          }
        }

        const fallback = r.mask_based_similarity
        return {
          ...r,
          is_match: fallback !== null && fallback !== undefined ? fallback >= threshold : r.is_match,
          _primary_metric: 'mask_based_similarity',
        }
      })

      return {
        ...page,
        results: processedResults,
      }
    })

    // 重要：PDF 最終階段常見「結果數量不變，但 overlay/heatmap 路徑補齊」。
    // 若用「可顯示結果數量」做 gate，會導致 completed 後 UI 停在空白/舊資料。
    // 以 results endpoint 作為真實來源：每次 polledPdfTaskResult 變更都同步到 UI。
    processedPages.sort((a, b) => (a?.page_index || 0) - (b?.page_index || 0))
    setPdfComparisonResults(processedPages)
  }, [polledPdfTaskResult, threshold, overlapWeight, pixelDiffPenaltyWeight, uniqueRegionPenaltyWeight])
  
  // 當輪詢結果更新時，同時更新狀態和結果顯示
  useEffect(() => {
    if (polledTaskResult) {
      // 更新任務狀態（從結果中提取狀態信息）
      const statusInfo = {
        status: polledTaskResult.status,
        progress: polledTaskResult.progress,
        progress_message: polledTaskResult.progress_message,
        total_count: polledTaskResult.total_count,
        success_count: polledTaskResult.success_count,
        error: polledTaskResult.error,
        created_at: polledTaskResult.created_at,
        started_at: polledTaskResult.started_at,
        completed_at: polledTaskResult.completed_at,
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

            // 新主指標：structure_similarity（由後端計算，對印泥深淺較不敏感）
            if (result.structure_similarity !== null && result.structure_similarity !== undefined) {
              const primary = result.structure_similarity
              const dynamicIsMatch = primary >= threshold
              return {
                ...result,
                mask_based_similarity: primary, // 前端沿用同一欄位顯示主分數
                is_match: dynamicIsMatch,
                _primary_metric: 'structure_similarity'
              }
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
                is_match: dynamicIsMatch,
                _primary_metric: 'mask_based_similarity'
              }
            }
            
            // 如果沒有 mask_statistics，使用後端返回的值作為備用
            const fallback = result.mask_based_similarity
            return {
              ...result,
              is_match: fallback !== null && fallback !== undefined ? fallback >= threshold : result.is_match,
              _primary_metric: 'mask_based_similarity'
            }
          })
          
          // 任務完成時強制更新結果，即使數量沒有增加
          // 否則只在結果數量增加時更新（降低 re-render 次數）
          const isTaskCompleted = polledTaskResult.status === 'completed'
          const shouldUpdate = isTaskCompleted || completedResults.length > lastCompletedCountRef.current
          
          if (shouldUpdate) {
            if (!isTaskCompleted) {
              lastCompletedCountRef.current = completedResults.length
            }
            setComparisonResults((prev) => {
              const prevArr = Array.isArray(prev) ? prev : []
              const keyOf = (r) => r?.seal_image_id || r?.seal_index
              const prevMap = new Map(prevArr.map(r => [keyOf(r), r]))

              for (const r of processedResults) {
                const k = keyOf(r)
                const old = prevMap.get(k)
                const unchanged = old && (
                  old.overlay1_path === r.overlay1_path &&
                  old.overlay2_path === r.overlay2_path &&
                  old.heatmap_path === r.heatmap_path &&
                  old.mask_based_similarity === r.mask_based_similarity &&
                  old.is_match === r.is_match &&
                  old.error === r.error
                )
                if (!unchanged) prevMap.set(k, r)
              }

              const merged = Array.from(prevMap.values())
              merged.sort((a, b) => (a.seal_index || 0) - (b.seal_index || 0))
              return merged
            })
          }
        } else if (polledTaskResult.status === 'completed') {
          // 如果任務已完成但沒有符合條件的結果，清空顯示（可能是所有結果都失敗了或沒有疊圖）
          setComparisonResults([])
        }
      } else if (polledTaskResult.status === 'completed' && !polledTaskResult.results) {
        // 任務完成但完全沒有 results 數據，清空顯示
        setComparisonResults([])
      }
    }
  }, [polledTaskResult, currentTaskUid, threshold, overlapWeight, pixelDiffPenaltyWeight, uniqueRegionPenaltyWeight])
  
  // 比對圖像1與多個印鑑（創建任務）
  const compareMutation = useMutation({
    mutationFn: ({ image1Id, sealImageIds, threshold, similaritySsimWeight, similarityTemplateWeight, pixelSimilarityWeight, histogramSimilarityWeight, overlapWeight, pixelDiffPenaltyWeight, uniqueRegionPenaltyWeight }) => 
      imageAPI.compareImage1WithSeals(image1Id, sealImageIds, threshold, similaritySsimWeight, similarityTemplateWeight, pixelSimilarityWeight, histogramSimilarityWeight, overlapWeight, pixelDiffPenaltyWeight, uniqueRegionPenaltyWeight),
    onSuccess: (taskData) => {
      // 保存任務 UID 並開始輪詢
      lastCompletedCountRef.current = 0
      setComparisonResults(null)
      setCurrentTaskUid(taskData.task_uid)
      setTaskStatus({
        status: taskData.status,
        progress: taskData.progress,
        progress_message: taskData.progress_message,
        total_count: taskData.total_count,
        success_count: taskData.success_count,
        error: taskData.error,
        created_at: taskData.created_at,
        started_at: taskData.started_at,
        completed_at: taskData.completed_at,
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
    if (image1EffectiveId) {
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
          confidence: clamp01(locationData.confidence) ?? 1.0
        }
        
        await updateSealLocation1Mutation.mutateAsync({
          imageId: image1EffectiveId,
          locationData: normalizedLocationData
        })

        // PDF：以 DB 為真。保存成功後立即 refetch 該 page image，確保主畫面預覽框同步更新
        if (image1?.is_pdf) {
          const refreshed = await imageAPI.get(image1EffectiveId)
          setImage1TemplatePage(refreshed)
          setImage1TemplatePageId(refreshed?.id || image1TemplatePageId)
        }

        // 模板更新後，舊的 PDF 全頁比對結果已不再適用，避免二次操作混用
        if (image2?.is_pdf) {
          resetPdfTaskUI()
        }

        // 等所有更新完成後再關閉，避免對話框關閉但主畫面仍是舊 state
        setShowSealDialog1(false)
      } catch (error) {
        console.error('更新印鑑位置失敗:', error)
        setSnackbar({
          open: true,
          message: `更新印鑑位置失敗：${error?.response?.data?.detail || error.message || '請重試'}`,
          severity: 'error'
        })
        throw error
      }
    }
  }

  // 處理圖像2多印鑑確認
  const handleMultipleSealsConfirm = async (seals) => {
    if (!uploadImage2Mutation.data?.id) return
    if (saveMultipleSealsMutation.isPending) return

    // PDF：保存到「選定頁」的 page image 上（僅用於預覽/手動確認；PDF 全頁比對仍會重新 auto detect）
    if (image2?.is_pdf) {
      if (!image2SelectedPageId) {
        setSnackbar({ open: true, message: '請先選擇要保存的 PDF 頁面', severity: 'warning' })
        return
      }
      try {
        await saveMultipleSealsMutation.mutateAsync({
          imageId: image2SelectedPageId,
          seals: seals
        })
        // 以 DB 為真：保存成功後 refetch 該 page image，確保主畫面預覽與後續流程一致
        const refreshed = await imageAPI.get(image2SelectedPageId)
        setImage2SelectedPage(refreshed)
        // 同步更新摘要（該頁顯示手動後的數量）
        const effectiveSeals = (refreshed?.multiple_seals && Array.isArray(refreshed.multiple_seals)) ? refreshed.multiple_seals : seals
        setPdfPageDetections2(prev => Array.isArray(prev)
          ? prev.map(p => String(p.page_image_id) === String(image2SelectedPageId)
              ? { ...p, detected: effectiveSeals.length > 0, seals: effectiveSeals, count: effectiveSeals.length, reason: effectiveSeals.length > 0 ? 'manual' : p.reason }
              : p
            )
          : prev
        )
        setShowSealDialog2(false)
      } catch (error) {
        console.error('保存 PDF 頁多印鑑位置失敗:', error)
        setSnackbar({ open: true, message: '保存失敗，請重試', severity: 'error' })
        throw error
      }
      return
    }

    // 一般圖片：維持既有行為
    setMultipleSeals(seals)
    try {
      await saveMultipleSealsMutation.mutateAsync({
        imageId: uploadImage2Mutation.data.id,
        seals: seals
      })
      setShowSealDialog2(false)
    } catch (error) {
      console.error('保存多印鑑位置失敗:', error)
      throw error
    }
  }

  // 處理清除
  const handleClearImage1 = () => {
    setImage1(null)
    setImage1TemplatePage(null)
    setImage1TemplatePageId(null)
    setImage1PreferredPageId(null)
    setSealDetectionResult1(null)
    setComparisonResults(null)
    // 清除 PDF 任務結果（模板清除後不應殘留）
    resetPdfTaskUI()
    lastCompletedCountRef.current = 0
    uploadImage1Mutation.reset()
  }

  const handleClearImage2 = () => {
    setImage2(null)
    setImage2SelectedPage(null)
    setImage2SelectedPageId(null)
    setImage2PreferredPageId(null)
    setMultipleSeals([])
    setPdfPageDetections2(null)
    setCroppedImageIds([])
    setComparisonResults(null)
    lastCompletedCountRef.current = 0
    setLastDetectionMaxSeals(null) // 重置追蹤狀態
    resetPdfTaskUI()
    uploadImage2Mutation.reset()
    // 重置 input value，允許重新選擇文件
    if (image2InputRef.current) {
      image2InputRef.current.value = ''
    }
  }

  // 處理編輯圖像1印鑑
  const handleEditImage1Seal = () => {
    if (image1EffectiveId) {
      setShowSealDialog1(true)
    } else if (image1?.is_pdf) {
      setSnackbar({ open: true, message: '請先等待 PDF 解析完成並選擇模板頁', severity: 'info' })
    }
  }

  // 處理編輯圖像2多印鑑
  const handleEditImage2Seals = async () => {
    if (uploadImage2Mutation.data?.id) {
      if (image2?.is_pdf) {
        if (!image2SelectedPageId) {
          setSnackbar({ open: true, message: '請先選擇要編輯的 PDF 頁面', severity: 'info' })
          return
        }
        setShowSealDialog2(true)
        return
      }
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

  // PDF 全頁比對：image1 模板頁 vs image2 全部頁
  const handlePdfCompare = async () => {
    if (!uploadImage1Mutation.data?.id || !uploadImage2Mutation.data?.id) {
      setSnackbar({ open: true, message: '請先上傳圖像1與圖像2', severity: 'warning' })
      return
    }
    if (!image2?.is_pdf) {
      setSnackbar({ open: true, message: '圖像2 需要是 PDF 才能使用全頁比對', severity: 'warning' })
      return
    }
    if (!image1EffectiveId || !image1HasSeal) {
      setSnackbar({ open: true, message: '圖像1 請先選定模板頁並標記印鑑位置', severity: 'warning' })
      return
    }

    try {
      // 重置 PDF 比對結果狀態（對齊 PNG/JPG 的處理）
      setPdfComparisonResults(null)
      const r = await imageAPI.comparePdf(image1EffectiveId, uploadImage2Mutation.data.id, {
        maxSeals,
        threshold,
        overlapWeight,
        pixelDiffPenaltyWeight,
        uniqueRegionPenaltyWeight
      })
      setPdfTaskUid(r.task_uid)
      setPdfTaskStatus({
        status: r.status,
        progress: r.progress,
        progress_message: r.progress_message,
        pages_total: r.pages_total,
        pages_done: r.pages_done,
      })
      // 立即查一次 status，確保 UI 能立刻進入輪詢/顯示頁數資訊（也避免背景分頁時輪詢延遲）
      try {
        const s = await imageAPI.getPdfTaskStatus(r.task_uid)
        setPdfTaskStatus(s)
      } catch (err) {
        console.warn('取得 PDF 任務狀態失敗（將依輪詢再同步）:', err)
      }
      setSnackbar({ open: true, message: `PDF 全頁比對已開始（UID: ${r.task_uid?.substring(0, 8)}...)`, severity: 'info' })
    } catch (e) {
      console.error(e)
      setSnackbar({ open: true, message: `PDF 全頁比對啟動失敗：${e?.response?.data?.detail || e.message}`, severity: 'error' })
    }
  }

  // 合併處理：保存、裁切、比對
  const handleCropAndCompare = async () => {
    // 1. 驗證前置條件
    if (image1?.is_pdf || image2?.is_pdf) {
      setSnackbar({
        open: true,
        message: 'PDF 請使用「PDF 全頁比對」流程',
        severity: 'info'
      })
      return
    }
    if (!uploadImage1Mutation.data?.id || !image1HasSeal) {
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
      {showAdvancedSettings && (
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
            {showMaxSealsSetting && showThresholdSetting && (
              <Typography variant="caption" color="text.secondary" sx={{ ml: 1 }}>
                （印鑑數量上限: {maxSeals} 個，相似度閾值: {Math.round(threshold * 100)}%）
              </Typography>
            )}
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          {/* 比對印鑑數量上限設定 */}
          {showMaxSealsSetting && (
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
          )}

          {/* 相似度閾值設定 */}
          {showThresholdSetting && (
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
          )}

          {/* Mask相似度權重參數設定 */}
          {showMaskWeightsSetting && (
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
        {showMaskWeightsSetting && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="caption" color={Math.abs(overlapWeight + pixelDiffPenaltyWeight + uniqueRegionPenaltyWeight - 1.0) < 0.01 ? 'success.main' : 'warning.main'}>
              權重總和: {(overlapWeight + pixelDiffPenaltyWeight + uniqueRegionPenaltyWeight).toFixed(2)} 
              {Math.abs(overlapWeight + pixelDiffPenaltyWeight + uniqueRegionPenaltyWeight - 1.0) < 0.01 ? ' ✓' : ' (建議調整為 1.0)'}
            </Typography>
          </Box>
        )}
          </Box>
          )}
        </AccordionDetails>
      </Accordion>
      )}

      <Grid container spacing={3}>
        {/* 圖像1區域 */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              圖像1（單印鑑）
            </Typography>
            <Box sx={{ mb: 2 }}>
              <input
                accept="image/*,application/pdf,.pdf"
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
              image={image1Effective}
              label={image1?.is_pdf ? '圖像1（PDF 模板頁）預覽' : '圖像1預覽'}
              onEdit={handleEditImage1Seal}
              showSealIndicator={true}
            />

            {image1?.is_pdf && (
              <PdfPagePicker
                pdfImage={image1}
                preferredPageImageId={image1PreferredPageId}
                label="模板頁"
                disabled={uploadImage1Mutation.isPending || isDetecting1}
                onPageImageLoaded={(pageImg) => {
                  // 防止舊快取回灌：若後到的資料比目前 state 更舊（updated_at 較早），不要覆寫
                  setImage1TemplatePage((prev) => {
                    if (!prev) return pageImg
                    const prevT = prev?.updated_at ? Date.parse(prev.updated_at) : NaN
                    const nextT = pageImg?.updated_at ? Date.parse(pageImg.updated_at) : NaN
                    if (Number.isFinite(prevT) && Number.isFinite(nextT) && nextT < prevT) {
                      return prev
                    }
                    return pageImg
                  })
                  // 只有「模板頁真的變更」才清除舊的 PDF 全頁比對任務狀態；
                  // 避免同一頁的 refetch/onPageImageLoaded 觸發把剛開始的任務清掉，導致 UI 看不到摘要/結果。
                  const nextTemplateId = pageImg?.id || null
                  const isPdfTaskTerminal = pdfTaskStatus?.status === 'completed' || pdfTaskStatus?.status === 'failed'
                  if (pdfTaskUid && isPdfTaskTerminal && String(nextTemplateId) !== String(image1TemplatePageId)) {
                    resetPdfTaskUI()
                  }
                  setImage1TemplatePageId(nextTemplateId)
                }}
              />
            )}

            {/* 編輯印鑑位置按鈕 */}
            {image1EffectiveId && (
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
                accept="image/*,application/pdf,.pdf"
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
              image={image2Effective}
              seals={image2EffectiveSeals}
              label={image2?.is_pdf ? '圖像2（PDF 單頁預覽）' : '圖像2預覽'}
              onSealClick={handleEditImage2Seals}
            />

            {image2?.is_pdf && (
              <Box sx={{ mt: 2 }}>
                <Alert severity="info">
                  圖像2為 PDF（共 {image2.pdf_page_count || image2.pages?.length || 0} 頁）。可先逐頁預覽/手動調整多印鑑框；但「PDF 全頁比對」仍會重新 auto detect。
                </Alert>
                <PdfPagePicker
                  pdfImage={image2}
                  preferredPageImageId={image2PreferredPageId}
                  label="預覽/編輯頁"
                  disabled={uploadImage2Mutation.isPending || isDetecting2}
                  onPageImageLoaded={(pageImg) => {
                    setImage2SelectedPage(pageImg)
                    setImage2SelectedPageId(pageImg?.id || null)
                  }}
                />
                {Array.isArray(pdfPageDetections2) && (
                  <Box sx={{ mt: 1 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      PDF 逐頁偵測摘要：
                    </Typography>
                    <Box sx={{ maxHeight: 180, overflow: 'auto', border: '1px solid', borderColor: 'divider', borderRadius: 1, p: 1 }}>
                      {pdfPageDetections2.map(p => (
                        <Typography key={`${p.page_index}-${p.page_image_id}`} variant="caption" sx={{ display: 'block' }}>
                          第 {p.page_index} 頁：{p.count} 個 {p.detected ? '' : '(未偵測到)'}
                        </Typography>
                      ))}
                    </Box>
                  </Box>
                )}
              </Box>
            )}

            {/* 操作按鈕 */}
            {image2?.is_pdf ? (
              <Box sx={{ mt: 2 }}>
                <Button
                  variant="outlined"
                  fullWidth
                  onClick={handleEditImage2Seals}
                  disabled={!image2SelectedPageId}
                  sx={{ mb: 1 }}
                >
                  編輯此頁多印鑑框
                </Button>
                <Button
                  variant="contained"
                  type="button"
                  fullWidth
                  onClick={handlePdfCompare}
                  disabled={!!pdfTaskUid && pdfTaskStatus?.status !== 'completed' && pdfTaskStatus?.status !== 'failed'}
                >
                  開始 PDF 全頁比對
                </Button>
                {pdfTaskStatus && (
                  <Alert severity={pdfTaskStatus.status === 'failed' ? 'error' : 'info'} sx={{ mt: 1 }}>
                    狀態：{pdfTaskStatus.status}{pdfTaskStatus.progress_message ? ` - ${pdfTaskStatus.progress_message}` : ''}
                  </Alert>
                )}
                {pdfTaskStatus?.status === 'processing' && pdfTaskStatus.progress >= 100 && 
                 (!polledPdfTaskResult?.results_by_page || polledPdfTaskResult.results_by_page.length === 0) && (
                  <Alert severity="warning" sx={{ mt: 1 }}>
                    任務進度已達 100%，但結果尚未完成。這可能是任務處理異常，請重新執行「開始 PDF 全頁比對」。
                  </Alert>
                )}
              </Box>
            ) : multipleSeals.length > 0 && (
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
                      !image1HasSeal ||
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
                        {(taskStatus.status === 'pending' || taskStatus.status === 'processing') && (
                          <Box sx={{ mt: 0.5 }}>
                            <Typography variant="body2" color="text.secondary">
                              {(() => {
                                const p = Number(taskStatus.progress)
                                const done = Number(taskStatus.success_count)
                                const total = Number(taskStatus.total_count)
                                const hasCounts = Number.isFinite(done) && Number.isFinite(total) && total >= 0
                                const countsText = hasCounts ? `${Math.max(0, done)}/${Math.max(0, total)}` : null

                                // 主指標：已完成/總數；輔助：百分比與訊息
                                if (!Number.isFinite(p)) {
                                  return `${countsText ? `進度：${countsText}` : '正在取得進度...'}${taskStatus.progress_message ? ` - ${taskStatus.progress_message}` : ''}`
                                }
                                return `${countsText ? `進度：${countsText}` : `進度：${p.toFixed(1)}%`}${taskStatus.progress_message ? ` - ${taskStatus.progress_message}` : ''}${countsText ? `（${p.toFixed(1)}%）` : ''}`
                              })()}
                            </Typography>
                            {(() => {
                              const p = Number(taskStatus.progress)
                              if (!Number.isFinite(p)) {
                                return <LinearProgress sx={{ mt: 0.75 }} />
                              }
                              const clamped = Math.max(0, Math.min(100, p))
                              return <LinearProgress variant="determinate" value={clamped} sx={{ mt: 0.75 }} />
                            })()}
                          </Box>
                        )}
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
      {showSimilarityHistogram && comparisonResults && comparisonResults.length > 0 && (
        <SimilarityHistogram
          results={comparisonResults}
        />
      )}

      {/* 顯示任務級別時間統計 */}
      {showTaskTimingStatistics && polledTaskResult?.task_timing && polledTaskResult.status === 'completed' && (
        <Box sx={{ mt: 4 }}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              任務時間統計
            </Typography>
            {(() => {
              const timing = polledTaskResult?.task_timing || {}
              const fmt = (v) => `${(typeof v === 'number' ? v : 0).toFixed(3)} 秒`

              const primaryDefs = [
                ['total_time', '總時間（服務層）'],
                ['parallel_processing_time', '並行處理時間（比對階段）'],
                ['average_seal_time', '平均每個印鑑時間（比對階段）'],
                ['avg_seal_total_time', '平均每顆印鑑總時間（results.timing.total）'],
              ]

              const avgStepDefs = [
                ['avg_load_images', '平均：載入圖像'],
                ['avg_remove_bg_image1', '平均：圖像1 去背景'],
                ['avg_remove_bg_align_image2', '平均：圖像2 去背景+對齊'],
                ['avg_save_aligned_images', '平均：保存對齊圖像'],
                ['avg_similarity_calculation', '平均：相似度計算'],
                ['avg_save_corrected_images', '平均：保存校正圖像'],
                ['avg_create_overlay', '平均：生成疊圖'],
                ['avg_calculate_mask_stats', '平均：計算 Mask 統計'],
                ['avg_create_heatmap', '平均：生成熱力圖'],
              ]

              const knownKeys = new Set([...primaryDefs, ...avgStepDefs].map(([k]) => k))
              const otherEntries = Object.entries(timing).filter(([k]) => !knownKeys.has(k))

              const renderGrid = (defs) => (
                <Grid container spacing={2}>
                  {defs.map(([k, label]) => (
                    <Grid key={k} item xs={12} sm={6} md={4}>
                      <Typography variant="body2" color="text.secondary">
                        {label}
                      </Typography>
                      <Typography variant="h6">
                        {timing[k] !== undefined ? fmt(timing[k]) : '—'}
                      </Typography>
                    </Grid>
                  ))}
                </Grid>
              )

              return (
                <Box>
                  {renderGrid(primaryDefs)}

                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                      平均步驟時間（僅統計成功的印鑑）
                    </Typography>
                    {renderGrid(avgStepDefs)}
                  </Box>

                  {otherEntries.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" sx={{ mb: 1 }}>
                        其他（task_timing）
                      </Typography>
                      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
                        {otherEntries.map(([k, v]) => (
                          <Box key={k} sx={{ display: 'flex', justifyContent: 'space-between', gap: 2 }}>
                            <Typography variant="body2" color="text.secondary" sx={{ fontFamily: 'monospace' }}>
                              {k}
                            </Typography>
                            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
                              {typeof v === 'number' ? fmt(v) : JSON.stringify(v)}
                            </Typography>
                          </Box>
                        ))}
                      </Box>
                    </Box>
                  )}
                </Box>
              )
            })()}
          </Paper>
        </Box>
      )}

      {/* 顯示比對結果 */}
      {comparisonResults && comparisonResults.length > 0 && (
        <Box sx={{ mt: 4 }}>
          <MultiSealComparisonResults 
            results={comparisonResults}
            image1Id={image1EffectiveId}
            similarityRange={null}
            threshold={threshold}
            overlapWeight={overlapWeight}
            pixelDiffPenaltyWeight={pixelDiffPenaltyWeight}
            uniqueRegionPenaltyWeight={uniqueRegionPenaltyWeight}
            calculateMaskBasedSimilarity={calculateMaskBasedSimilarity}
          />
        </Box>
      )}

      {/* 任務完成但沒有符合條件的結果時顯示提示 */}
      {taskStatus?.status === 'completed' && 
       (!comparisonResults || comparisonResults.length === 0) && 
       polledTaskResult && (
        <Box sx={{ mt: 4 }}>
          <Alert severity="info">
            <Typography variant="body2" gutterBottom>
              任務已完成，但沒有可顯示的比對結果。
            </Typography>
            {polledTaskResult.total_count !== undefined && polledTaskResult.total_count > 0 && (
              <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                總共處理了 {polledTaskResult.total_count} 個印鑑，成功 {polledTaskResult.success_count || 0} 個。
                {polledTaskResult.results && polledTaskResult.results.length > 0 && (
                  <span> 但所有結果都缺少疊圖數據（overlay1_path/overlay2_path）。</span>
                )}
              </Typography>
            )}
            {polledTaskResult.error && (
              <Typography variant="caption" display="block" sx={{ mt: 0.5, color: 'error.main' }}>
                錯誤：{polledTaskResult.error}
              </Typography>
            )}
          </Alert>
        </Box>
      )}

      {/* 顯示 PDF 全頁比對結果（攤平顯示） */}
      {image2?.is_pdf && (pdfTaskUid || pdfTaskStatus || polledPdfTaskResult) && (
        <Box sx={{ mt: 4 }}>
          {(() => {
            const statusFromStatus = pdfTaskStatus?.status
            const statusFromResult = polledPdfTaskResult?.status
            const isTerminal = (s) => s === 'completed' || s === 'failed'
            const effectiveStatus =
              isTerminal(statusFromStatus) ? statusFromStatus
              : isTerminal(statusFromResult) ? statusFromResult
              : (statusFromResult || statusFromStatus || '—')

            const pagesTotalRaw =
              polledPdfTaskResult?.pages_total ?? pdfTaskStatus?.pages_total ?? polledPdfTaskStatus?.pages_total
            const pagesDoneRaw =
              polledPdfTaskResult?.pages_done ?? pdfTaskStatus?.pages_done ?? polledPdfTaskStatus?.pages_done

            const pagesTotal = Number(pagesTotalRaw)
            const pagesDone = Number(pagesDoneRaw)
            const hasCounts =
              Number.isFinite(pagesTotal) && pagesTotal > 0 && Number.isFinite(pagesDone) && pagesDone >= 0

            const progressPct = hasCounts ? (pagesDone / pagesTotal) * 100 : Number(pdfTaskStatus?.progress)
            const hasProgress = Number.isFinite(progressPct)

            const rawPages = Array.isArray(pdfComparisonResults)
              ? pdfComparisonResults
              : (Array.isArray(polledPdfTaskResult?.results_by_page) ? polledPdfTaskResult.results_by_page : [])

            const pagesWithDetected = rawPages.filter(p => !!p?.detected).length
            const pagesWithVisibleResults = rawPages.filter(p => Array.isArray(p?.results) && p.results.length > 0).length
            const visibleResultsCount = rawPages.reduce((sum, p) => sum + (Array.isArray(p?.results) ? p.results.length : 0), 0)
            const pagesWithReason = rawPages.filter(p => !p?.detected && p?.reason).length

            const waitingFinal =
              effectiveStatus === 'completed' &&
              (!polledPdfTaskResult || !Array.isArray(polledPdfTaskResult.results_by_page) || polledPdfTaskResult.results_by_page.length === 0)

            return (
              <Paper sx={{ p: 2 }}>
                <Typography variant="subtitle1" gutterBottom>
                  PDF 任務摘要
            </Typography>
                <Typography variant="body2" color="text.secondary">
                  Job: {pdfTaskUid || polledPdfTaskResult?.task_uid || '—'}
                  </Typography>
                <Typography variant="body2" sx={{ mt: 0.5 }}>
                  狀態：{effectiveStatus}
                  {hasCounts ? `（頁面：${Math.max(0, pagesDone)}/${Math.max(0, pagesTotal)}）` : ''}
                  </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                  {`頁面統計：有偵測 ${pagesWithDetected} 頁｜有可顯示結果 ${pagesWithVisibleResults} 頁｜可顯示結果總數 ${visibleResultsCount}｜無偵測原因 ${pagesWithReason} 頁`}
                </Typography>

                <Box sx={{ mt: 1 }}>
                  {hasProgress ? (
                    <LinearProgress variant="determinate" value={Math.max(0, Math.min(100, progressPct))} />
                  ) : (
                    <LinearProgress />
                  )}
                </Box>

                {waitingFinal && (
                  <Alert severity="info" sx={{ mt: 2 }}>
                    任務顯示完成，但尚未取得最終結果資料（正在重新查詢結果…）
                  </Alert>
                )}
              </Paper>
            )
          })()}

          {(() => {
            const statusFromStatus = pdfTaskStatus?.status
            const statusFromResult = polledPdfTaskResult?.status
            const isTerminal = (s) => s === 'completed' || s === 'failed'
            const effectiveStatus =
              isTerminal(statusFromStatus) ? statusFromStatus
              : isTerminal(statusFromResult) ? statusFromResult
              : (statusFromResult || statusFromStatus)

            if (effectiveStatus !== 'completed') return null

            if (!polledPdfTaskResult) {
              return (
                <Alert severity="info" sx={{ mt: 2 }}>
                  任務已完成，正在取得最終結果…
          </Alert>
              )
            }

            if (Array.isArray(polledPdfTaskResult.results_by_page) && polledPdfTaskResult.results_by_page.length === 0) {
              return (
                <Alert severity="warning" sx={{ mt: 2 }}>
                  PDF 比對任務已完成，但後端回傳 results_by_page 為空。請檢查後端日誌或重新執行比對。
                </Alert>
              )
            }

            return null
          })()}
        </Box>
      )}

      {image2?.is_pdf && pdfTaskUid && (
        <Box sx={{ mt: 4 }}>
          {polledPdfTaskResult?.status === 'processing' && (
            <Alert severity="info" sx={{ mb: 2 }}>
              任務處理完成，結果已可用（狀態更新中）
            </Alert>
          )}

          {/* PDF Histogram（全頁合併） */}
          {showSimilarityHistogram && (() => {
            const rawPages = Array.isArray(pdfComparisonResults) ? pdfComparisonResults : []
            const allResults = rawPages.flatMap(p => Array.isArray(p?.results) ? p.results : [])
            if (!allResults || allResults.length === 0) return null
            return (
              <SimilarityHistogram
                results={allResults}
                selectedRange={pdfGlobalFilter.similarityRange}
                onRangeSelect={(range) => setPdfGlobalFilter((prev) => ({ ...prev, similarityRange: range }))}
              />
            )
          })()}

          {/* PDF 全頁共用篩選器（只顯示一套） */}
          <Paper sx={{ p: 2, mt: 2, mb: 2 }}>
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
              <TextField
                size="small"
                placeholder="搜尋印鑑索引..."
                value={pdfGlobalFilter.searchText}
                onChange={(e) => setPdfGlobalFilter((prev) => ({ ...prev, searchText: e.target.value }))}
                inputProps={{ 'data-testid': 'pdf-global-filter-search' }}
                InputProps={{
                  startAdornment: <SearchIcon sx={{ mr: 1, color: 'text.secondary' }} />,
                }}
                sx={{ flex: 1 }}
              />

              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TextField
                  size="small"
                  type="number"
                  placeholder="最小%"
                  value={pdfGlobalFilter.minSimilarity}
                  onChange={(e) => {
                    const val = e.target.value
                    if (val === '' || (parseFloat(val) >= 0 && parseFloat(val) <= 100)) {
                      setPdfGlobalFilter((prev) => ({ ...prev, minSimilarity: val }))
                    }
                  }}
                  inputProps={{ min: 0, max: 100, step: 0.1 }}
                  sx={{ width: 110 }}
                  label="最小相似度"
                />
                <Typography variant="body2" color="text.secondary">
                  ~
                </Typography>
                <TextField
                  size="small"
                  type="number"
                  placeholder="最大%"
                  value={pdfGlobalFilter.maxSimilarity}
                  onChange={(e) => {
                    const val = e.target.value
                    if (val === '' || (parseFloat(val) >= 0 && parseFloat(val) <= 100)) {
                      setPdfGlobalFilter((prev) => ({ ...prev, maxSimilarity: val }))
                    }
                  }}
                  inputProps={{ min: 0, max: 100, step: 0.1 }}
                  sx={{ width: 110 }}
                  label="最大相似度"
                />
              </Box>

              <FormControl size="small" sx={{ minWidth: 150 }}>
                <InputLabel>匹配狀態</InputLabel>
                <Select
                  value={pdfGlobalFilter.matchFilter}
                  label="匹配狀態"
                  onChange={(e) => setPdfGlobalFilter((prev) => ({ ...prev, matchFilter: e.target.value }))}
                >
                  <MenuItem value="all">全部</MenuItem>
                  <MenuItem value="match">匹配</MenuItem>
                  <MenuItem value="no-match">不匹配</MenuItem>
                </Select>
              </FormControl>

              <FormControl size="small" sx={{ minWidth: 150 }}>
                <InputLabel>排序方式</InputLabel>
                <Select
                  value={pdfGlobalFilter.sortBy}
                  label="排序方式"
                  onChange={(e) => setPdfGlobalFilter((prev) => ({ ...prev, sortBy: e.target.value }))}
                >
                  <MenuItem value="index-asc">印鑑索引 ↑</MenuItem>
                  <MenuItem value="index-desc">印鑑索引 ↓</MenuItem>
                  <MenuItem value="similarity-asc">相似度 ↑</MenuItem>
                  <MenuItem value="similarity-desc">相似度 ↓</MenuItem>
                </Select>
              </FormControl>
            </Stack>
          </Paper>

          {(() => {
            const pagesTotalRaw =
              polledPdfTaskStatus?.pages_total ?? pdfTaskStatus?.pages_total ?? polledPdfTaskResult?.pages_total
            const pagesTotal = Number(pagesTotalRaw)

            const pagesFromProcessed = Array.isArray(pdfComparisonResults) ? pdfComparisonResults : []
            const pagesFromResult = Array.isArray(polledPdfTaskResult?.results_by_page) ? polledPdfTaskResult.results_by_page : []
            let pages = pagesFromProcessed.length > 0 ? [...pagesFromProcessed] : [...pagesFromResult]

            if (pages.length === 0 && Number.isFinite(pagesTotal) && pagesTotal > 0) {
              // 沒拿到 results_by_page 前也先把頁級 UI 架起來（避免 UI 空白）
              pages = Array.from({ length: pagesTotal }, (_, idx) => ({
                page_index: idx + 1,
                page_image_id: `pending-${idx + 1}`,
                detected: false,
                count: 0,
                results: [],
                reason: '等待結果',
              }))
            }

            pages.sort((a, b) => (a?.page_index || 0) - (b?.page_index || 0))

            const normalizeResults = (page) => (Array.isArray(page?.results) ? page.results : [])
            const scoreOf = (x) => (x?.mask_based_similarity ?? x?.structure_similarity ?? x?.similarity ?? 0)

            return (
              <Box sx={{ mt: 2 }}>
                {pages.map((p) => {
                  const normalized = normalizeResults(p)
                  const best = normalized.length > 0 ? Math.max(...normalized.map(x => scoreOf(x))) : 0
                  const suffix = p.detected
                    ? `（${p.count} 個，最佳 ${(best * 100).toFixed(1)}%）`
                    : (p.reason ? `（${p.reason}）` : '（未偵測到）')

                  return (
                    <Box key={`${p.page_index}-${p.page_image_id}`} sx={{ mt: 3 }}>
                      <Typography variant="subtitle2">
                        第 {p.page_index} 頁 {suffix}
                      </Typography>

                      {normalized.length > 0 ? (
                        <Box sx={{ mt: 2 }}>
                          <MultiSealComparisonResults
                            results={normalized}
                            image1Id={image1EffectiveId}
                            threshold={threshold}
                            overlapWeight={overlapWeight}
                            pixelDiffPenaltyWeight={pixelDiffPenaltyWeight}
                            uniqueRegionPenaltyWeight={uniqueRegionPenaltyWeight}
                            calculateMaskBasedSimilarity={calculateMaskBasedSimilarity}
                            controlsMode="external"
                            filterState={pdfGlobalFilter}
                            onFilterStateChange={setPdfGlobalFilter}
                            hideControls={true}
                          />
                        </Box>
                      ) : (
                        <Alert severity="info" sx={{ mt: 2 }}>
                          第 {p.page_index} 頁沒有比對結果：{p.reason || '未檢測到印鑑'}
                        </Alert>
                      )}
                    </Box>
                  )
                })}
              </Box>
            )
          })()}
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
          {image1EffectiveId && (
            <SealDetectionBox
              // 重要：避免 Dialog 關閉/重開或保存後 state 沒刷新而顯示舊 bbox
              // - image1Effective.updated_at 會在 PUT /seal-location 後更新
              // - 以 key 強制 remount，確保初始值一定來自最新 props
              key={`${image1EffectiveId}:${image1Effective?.updated_at || ''}`}
              imageId={image1EffectiveId}
              initialBbox={image1Effective?.seal_bbox || sealDetectionResult1?.bbox || null}
              initialCenter={image1Effective?.seal_center || sealDetectionResult1?.center || null}
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
          {(image2?.is_pdf ? image2SelectedPageId : uploadImage2Mutation.data?.id) && (
            <MultiSealDetectionBox
              imageId={image2?.is_pdf ? image2SelectedPageId : uploadImage2Mutation.data.id}
              initialSeals={image2EffectiveSeals}
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

