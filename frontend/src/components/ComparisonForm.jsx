import React, { useState, useEffect } from 'react'
import { Button, Box, Alert, Dialog, DialogTitle, DialogContent, Grid, Typography, Divider, ToggleButton, ToggleButtonGroup, CircularProgress } from '@mui/material'
import { ContentCopy as ContentCopyIcon, ViewAgenda as ViewAgendaIcon } from '@mui/icons-material'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { imageAPI } from '../services/api'
import SealDetectionBox from './SealDetectionBox'
import ImagePreview from './ImagePreview'
import BatchSealAdjustment from './BatchSealAdjustment'
import PdfPagePicker from './PdfPagePicker'
import { useFeatureFlag, FEATURE_FLAGS } from '../config/featureFlags'

function ComparisonForm({ onSubmit }) {
  const queryClient = useQueryClient()
  const [image1, setImage1] = useState(null)
  const [image2, setImage2] = useState(null)
  // 圖像1為 PDF 時：使用者選定的模板頁（page image）
  const [image1TemplatePage, setImage1TemplatePage] = useState(null)
  const [image1TemplatePageId, setImage1TemplatePageId] = useState(null)
  const [image1PreferredPageId, setImage1PreferredPageId] = useState(null)
  const [enableRotation, setEnableRotation] = useState(true)
  const [showSealDialog, setShowSealDialog] = useState(false)
  const [sealDetectionResult, setSealDetectionResult] = useState(null)
  const [isDetecting, setIsDetecting] = useState(false)
  const [showSealDialog2, setShowSealDialog2] = useState(false)
  const [sealDetectionResult2, setSealDetectionResult2] = useState(null)
  const [isDetecting2, setIsDetecting2] = useState(false)
  const [batchMode, setBatchMode] = useState(false)
  const [showBatchDialog, setShowBatchDialog] = useState(false)
  
  // 功能開關
  const showBatchSealAdjustment = useFeatureFlag(FEATURE_FLAGS.BATCH_SEAL_ADJUSTMENT)

  const image1Effective = image1?.is_pdf ? image1TemplatePage : image1
  const image1EffectiveId = image1?.is_pdf ? image1TemplatePageId : image1?.id
  const image1HasSeal = !!image1Effective?.seal_bbox

  const uploadImage1Mutation = useMutation({
    mutationFn: imageAPI.upload,
    onSuccess: async (data) => {
      setImage1(data)
      setImage1TemplatePage(null)
      setImage1TemplatePageId(null)
      setImage1PreferredPageId(null)
      // 自動檢測印鑑
      setIsDetecting(true)
      try {
        const detectionResult = await imageAPI.detectSeal(data.id)
        setSealDetectionResult(detectionResult)

        // PDF：後端會回傳最佳頁 page_image_id/page_index；前端跳到該頁並讓使用者手動微調
        if (data?.is_pdf) {
          const pageId = detectionResult?.page_image_id || data?.pages?.[0]?.id || null
          setImage1PreferredPageId(pageId)

          if (pageId) {
            const pageImage = await imageAPI.get(pageId)
            setImage1TemplatePage(pageImage)
            setImage1TemplatePageId(pageId)
            setShowSealDialog(true)
          } else {
            // 沒有分頁資訊時，無法提供框選（這應該很少發生）
            setShowSealDialog(false)
          }
          return
        }
        
        // 檢查檢測是否成功（detected === true 且 bbox 存在）
        if (detectionResult.detected === true && detectionResult.bbox) {
          // 檢測成功，自動確認並保存結果
          try {
            // 構建 locationData（與 handleSealConfirm 使用相同的格式）
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
            
            // 自動保存檢測結果（onSuccess 會自動更新 image1 state 和關閉對話框）
            await updateSealLocationMutation.mutateAsync({
              imageId: data.id,
              locationData: normalizedLocationData
            })
            
            // 注意：不顯示對話框由 updateSealLocationMutation 的 onSuccess 處理
          } catch (error) {
            console.error('自動確認失敗:', error)
            // 如果自動確認失敗，回退到顯示對話框讓用戶手動操作
            setShowSealDialog(true)
          }
        } else {
          // 檢測失敗，顯示對話框讓用戶手動框選
          setShowSealDialog(true)
        }
      } catch (error) {
        console.error('檢測失敗:', error)
        // 檢測異常時設置為失敗狀態，顯示對話框讓用戶手動框選
        setSealDetectionResult({ detected: false, bbox: null, center: null })
        setShowSealDialog(true)
      } finally {
        setIsDetecting(false)
      }
    },
  })

  const uploadImage2Mutation = useMutation({
    mutationFn: imageAPI.upload,
    onSuccess: async (data) => {
      setImage2(data)
      // 自動檢測印鑑
      setIsDetecting2(true)
      try {
        const detectionResult = await imageAPI.detectSeal(data.id)
        setSealDetectionResult2(detectionResult)
        // 無論檢測是否成功，都顯示對話框
        setShowSealDialog2(true)
      } catch (error) {
        console.error('檢測失敗:', error)
        // 檢測失敗時設置為 null，SealDetectionBox 會使用默認值
        setSealDetectionResult2({ detected: false, bbox: null, center: null })
        // 仍然顯示對話框，讓用戶手動框選
        setShowSealDialog2(true)
      } finally {
        setIsDetecting2(false)
      }
    },
  })

  const updateSealLocationMutation = useMutation({
    mutationFn: ({ imageId, locationData }) => imageAPI.updateSealLocation(imageId, locationData),
    onSuccess: (data, variables) => {
      // 根據 imageId 判斷關閉哪個對話框
      if (image1?.is_pdf) {
        if (variables.imageId === image1TemplatePageId) {
          setShowSealDialog(false)
          setImage1TemplatePage(data)
        }
        return
      }

      if (variables.imageId === uploadImage1Mutation.data?.id) {
        setShowSealDialog(false)
        // 更新 image1 狀態
        setImage1(data)
        // 更新 mutation 的 data，確保按鈕狀態正確
        // 通過直接修改 mutation 的內部狀態
        if (uploadImage1Mutation.data) {
          Object.assign(uploadImage1Mutation.data, data)
        }
      }
    },
  })
  
  const updateSealLocationMutation2 = useMutation({
    mutationFn: ({ imageId, locationData }) => imageAPI.updateSealLocation(imageId, locationData),
    onSuccess: (data) => {
      setShowSealDialog2(false)
      setImage2(data)
      // 更新 mutation 的 data，確保按鈕狀態正確
      if (uploadImage2Mutation.data) {
        Object.assign(uploadImage2Mutation.data, data)
      }
    },
  })

  const handleImage1Change = (e) => {
    const file = e.target.files[0]
    if (file) {
      uploadImage1Mutation.mutate(file)
    }
  }

  const handleImage2Change = (e) => {
    const file = e.target.files[0]
    if (file) {
      uploadImage2Mutation.mutate(file)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (
      image1EffectiveId &&
      uploadImage2Mutation.data?.id
    ) {
      onSubmit(
        image1EffectiveId,
        uploadImage2Mutation.data.id,
        enableRotation
      )
    }
  }

  const handleSealConfirm = async (locationData) => {
    if (image1EffectiveId) {
      try {
        // 確保數據格式正確
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
        
        await updateSealLocationMutation.mutateAsync({
          imageId: image1EffectiveId,
          locationData: normalizedLocationData
        })
      } catch (error) {
        console.error('更新印鑑位置失敗:', error)
        alert('更新印鑑位置失敗，請重試')
      }
    }
  }

  const handleSealCancel = () => {
    setShowSealDialog(false)
  }

  const handleSealConfirm2 = async (locationData) => {
    if (uploadImage2Mutation.data?.id) {
      try {
        // 確保數據格式正確
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
        
        await updateSealLocationMutation2.mutateAsync({
          imageId: uploadImage2Mutation.data.id,
          locationData: normalizedLocationData
        })
      } catch (error) {
        console.error('更新印鑑位置失敗:', error)
        alert('更新印鑑位置失敗，請重試')
      }
    }
  }

  const handleSealCancel2 = () => {
    setShowSealDialog2(false)
  }

  // 處理使用同一張照片
  const handleUseSameImage = async () => {
    if (!uploadImage1Mutation.data) {
      alert('請先上傳圖像1')
      return
    }

    if (!image1HasSeal) {
      alert('請先為圖像1標記印鑑位置')
      return
    }

    try {
      // 獲取圖像1的文件並重新上傳作為圖像2
      const image1Data = image1Effective || uploadImage1Mutation.data
      const image1FileUrl = imageAPI.getFile(image1Data.id)
      
      // 從 URL 獲取文件並重新上傳
      const response = await fetch(image1FileUrl)
      const blob = await response.blob()
      const file = new File([blob], image1Data.filename || 'image.jpg', { type: blob.type })
      
      // 上傳作為圖像2
      uploadImage2Mutation.mutate(file, {
        onSuccess: async (uploadedImage2) => {
          // 複製圖像1的印鑑位置到圖像2
          try {
            await updateSealLocationMutation2.mutateAsync({
              imageId: uploadedImage2.id,
              locationData: {
                bbox: image1Data.seal_bbox,
                center: image1Data.seal_center,
                confidence: image1Data.seal_confidence || 1.0
              }
            })
          } catch (error) {
            console.error('複製印鑑位置失敗:', error)
            // 即使複製失敗，也顯示對話框讓用戶手動確認
            setShowSealDialog2(true)
          }
        }
      })
    } catch (error) {
      console.error('使用同一張照片失敗:', error)
      alert('操作失敗，請重試')
    }
  }

  // 處理清除圖像
  const handleClearImage1 = () => {
    setImage1(null)
    setImage1TemplatePage(null)
    setImage1TemplatePageId(null)
    setImage1PreferredPageId(null)
    setSealDetectionResult(null)
    uploadImage1Mutation.reset()
  }

  const handleClearImage2 = () => {
    setImage2(null)
    setSealDetectionResult2(null)
    uploadImage2Mutation.reset()
  }

  // 處理編輯圖像1的印鑑位置
  const handleEditImage1Seal = () => {
    if (image1EffectiveId) {
      if (batchMode && uploadImage2Mutation.data?.id) {
        setShowBatchDialog(true)
      } else {
        setShowSealDialog(true)
      }
    }
  }

  // 處理編輯圖像2的印鑑位置
  const handleEditImage2Seal = () => {
    if (uploadImage2Mutation.data?.id) {
      if (batchMode) {
        setShowBatchDialog(true)
      } else {
        setShowSealDialog2(true)
      }
    }
  }

  // 處理批量確認
  const handleBatchConfirm = async (locationData) => {
    try {
      // 確認圖像1
      if (locationData.image1 && image1EffectiveId) {
        const normalizedLocationData1 = {
          bbox: locationData.image1.bbox ? {
            x: Math.round(locationData.image1.bbox.x),
            y: Math.round(locationData.image1.bbox.y),
            width: Math.round(locationData.image1.bbox.width),
            height: Math.round(locationData.image1.bbox.height)
          } : null,
          center: locationData.image1.center ? {
            center_x: Math.round(locationData.image1.center.center_x),
            center_y: Math.round(locationData.image1.center.center_y),
            radius: typeof locationData.image1.center.radius === 'number' ? locationData.image1.center.radius : 0
          } : null,
          confidence: 1.0
        }
        await updateSealLocationMutation.mutateAsync({
          imageId: image1EffectiveId,
          locationData: normalizedLocationData1
        })
      }

      // 確認圖像2
      if (locationData.image2 && uploadImage2Mutation.data?.id) {
        const normalizedLocationData2 = {
          bbox: locationData.image2.bbox ? {
            x: Math.round(locationData.image2.bbox.x),
            y: Math.round(locationData.image2.bbox.y),
            width: Math.round(locationData.image2.bbox.width),
            height: Math.round(locationData.image2.bbox.height)
          } : null,
          center: locationData.image2.center ? {
            center_x: Math.round(locationData.image2.center.center_x),
            center_y: Math.round(locationData.image2.center.center_y),
            radius: typeof locationData.image2.center.radius === 'number' ? locationData.image2.center.radius : 0
          } : null,
          confidence: 1.0
        }
        await updateSealLocationMutation2.mutateAsync({
          imageId: uploadImage2Mutation.data.id,
          locationData: normalizedLocationData2
        })
      }

      setShowBatchDialog(false)
    } catch (error) {
      console.error('批量確認失敗:', error)
      alert('批量確認失敗，請重試')
    }
  }

  const handleBatchCancel = () => {
    setShowBatchDialog(false)
  }

  // 處理複製圖像1的印鑑位置到圖像2
  const handleCopySealLocation = async () => {
    if (!uploadImage1Mutation.data || !uploadImage2Mutation.data) {
      alert('請先上傳兩個圖像')
      return
    }

    // 使用 mutation data 或 state，確保獲取最新的印鑑位置
    const image1Data = image1Effective || image1 || uploadImage1Mutation.data
    if (!image1Data?.seal_bbox) {
      alert('圖像1還沒有標記印鑑位置')
      return
    }

    try {
      await updateSealLocationMutation2.mutateAsync({
        imageId: uploadImage2Mutation.data.id,
        locationData: {
          bbox: image1Data.seal_bbox,
          center: image1Data.seal_center,
          confidence: image1Data.seal_confidence || 1.0
        }
      })
    } catch (error) {
      console.error('複製印鑑位置失敗:', error)
      alert('複製印鑑位置失敗，請重試')
    }
  }

  return (
    <Box component="form" onSubmit={handleSubmit}>
      <Grid container spacing={2} sx={{ mb: 2 }}>
        {/* 圖像1區域 */}
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle1" gutterBottom fontWeight="bold">
              圖像1
            </Typography>
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
                disabled={uploadImage1Mutation.isPending || isDetecting}
                sx={{ mb: 1 }}
                startIcon={
                  (uploadImage1Mutation.isPending || isDetecting) && (
                    <CircularProgress size={16} />
                  )
                }
              >
                {uploadImage1Mutation.isPending
                  ? '上傳中...'
                  : isDetecting
                  ? '檢測印鑑中...'
                  : '選擇圖像1'}
              </Button>
            </label>
            {(uploadImage1Mutation.isPending || isDetecting) && (
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
          
          {/* 圖像預覽 */}
          <ImagePreview
            image={image1Effective}
            label={image1?.is_pdf ? `圖像1（PDF 模板頁）預覽` : "圖像1預覽"}
            onEdit={handleEditImage1Seal}
            showSealIndicator={true}
          />
          {image1?.is_pdf && (
            <PdfPagePicker
              pdfImage={image1}
              preferredPageImageId={image1PreferredPageId}
              label="模板頁"
              disabled={isDetecting || uploadImage1Mutation.isPending}
              onPageImageLoaded={(pageImg) => {
                setImage1TemplatePage(pageImg)
                setImage1TemplatePageId(pageImg?.id || null)
              }}
            />
          )}
        </Grid>

        {/* 圖像2區域 */}
        <Grid item xs={12} md={6}>
          <Box sx={{ mb: 2 }}>
            <Typography variant="subtitle1" gutterBottom fontWeight="bold">
              圖像2
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
              <input
                accept="image/*,application/pdf,.pdf"
                style={{ display: 'none' }}
                id="image2-upload"
                type="file"
                onChange={handleImage2Change}
              />
              <label htmlFor="image2-upload" style={{ flex: 1 }}>
                <Button
                  variant="outlined"
                  component="span"
                  fullWidth
                  disabled={
                    uploadImage2Mutation.isPending || 
                    isDetecting2 || 
                    !image1HasSeal
                  }
                  title={
                    !image1HasSeal
                      ? '請先完成圖像1的印章位置確認'
                      : undefined
                  }
                  startIcon={
                    (uploadImage2Mutation.isPending || isDetecting2) && (
                      <CircularProgress size={16} />
                    )
                  }
                >
                  {uploadImage2Mutation.isPending
                    ? '上傳中...'
                    : isDetecting2
                    ? '檢測印鑑中...'
                    : '選擇圖像2'}
                </Button>
              </label>
              {uploadImage1Mutation.data && (
                <Button
                  variant="outlined"
                  startIcon={<ContentCopyIcon />}
                  onClick={handleUseSameImage}
                  disabled={!image1HasSeal}
                  title="使用圖像1作為圖像2（需要先標記圖像1的印鑑位置）"
                >
                  使用同一張
                </Button>
              )}
            </Box>
            {!image1HasSeal && uploadImage1Mutation.data && (
              <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                請先完成圖像1的印章位置確認
              </Typography>
            )}
            {(uploadImage2Mutation.isPending || isDetecting2) && (
              <Alert severity="info" sx={{ mt: 1 }}>
                {uploadImage2Mutation.isPending ? '正在上傳圖像...' : '正在自動檢測印章位置...'}
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
          
          {/* 圖像預覽 */}
          <ImagePreview
            image={image2}
            label="圖像2預覽"
            onEdit={handleEditImage2Seal}
            showSealIndicator={true}
          />
          
          {/* 複製印鑑位置按鈕 */}
          {uploadImage1Mutation.data && 
           uploadImage2Mutation.data && 
           image1HasSeal && 
           !(image2?.seal_bbox || uploadImage2Mutation.data?.seal_bbox) && (
            <Button
              variant="outlined"
              size="small"
              startIcon={<ContentCopyIcon />}
              onClick={handleCopySealLocation}
              fullWidth
              sx={{ mt: 1 }}
              title="複製圖像1的印鑑位置到圖像2"
            >
              複製圖像1的印鑑位置
            </Button>
          )}
        </Grid>
      </Grid>

      <Divider sx={{ my: 2 }} />

      <Button
        type="submit"
        variant="contained"
        fullWidth
        size="large"
        disabled={
          !uploadImage1Mutation.data ||
          !uploadImage2Mutation.data ||
          uploadImage1Mutation.isPending ||
          uploadImage2Mutation.isPending ||
          showSealDialog ||
          showSealDialog2 ||
          showBatchDialog ||
          !image1HasSeal ||
          !(image2?.seal_bbox || uploadImage2Mutation.data?.seal_bbox)
        }
      >
        開始比對
      </Button>

      <Dialog
        open={showSealDialog}
        onClose={() => {}} // 不允許點擊外部關閉
        disableEscapeKeyDown={true} // 不允許按 ESC 關閉
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>調整圖像1印鑑位置</DialogTitle>
        <DialogContent>
          {image1EffectiveId && (
            <SealDetectionBox
              imageId={image1EffectiveId}
              initialBbox={image1Effective?.seal_bbox || sealDetectionResult?.bbox || null}
              initialCenter={image1Effective?.seal_center || sealDetectionResult?.center || null}
              onConfirm={handleSealConfirm}
              onCancel={handleSealCancel}
            />
          )}
        </DialogContent>
      </Dialog>

      <Dialog
        open={showSealDialog2}
        onClose={() => {}} // 不允許點擊外部關閉
        disableEscapeKeyDown={true} // 不允許按 ESC 關閉
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>調整圖像2印鑑位置</DialogTitle>
        <DialogContent>
          {uploadImage2Mutation.data?.id && (
            <SealDetectionBox
              imageId={uploadImage2Mutation.data.id}
              initialBbox={sealDetectionResult2?.bbox || uploadImage2Mutation.data?.seal_bbox || null}
              initialCenter={sealDetectionResult2?.center || uploadImage2Mutation.data?.seal_center || null}
              onConfirm={handleSealConfirm2}
              onCancel={handleSealCancel2}
            />
          )}
        </DialogContent>
      </Dialog>

      {/* 批量調整對話框 */}
      {showBatchSealAdjustment && (
        <Dialog
          open={showBatchDialog}
          onClose={() => {}} // 不允許點擊外部關閉
          disableEscapeKeyDown={true} // 不允許按 ESC 關閉
          maxWidth="lg"
          fullWidth
        >
          <DialogTitle>批量調整印鑑位置</DialogTitle>
          <DialogContent>
            {image1EffectiveId && uploadImage2Mutation.data?.id && (
              <BatchSealAdjustment
                image1Id={image1EffectiveId}
                image2Id={uploadImage2Mutation.data.id}
                image1InitialBbox={image1Effective?.seal_bbox || sealDetectionResult?.bbox || null}
                image1InitialCenter={image1Effective?.seal_center || sealDetectionResult?.center || null}
                image2InitialBbox={sealDetectionResult2?.bbox || uploadImage2Mutation.data?.seal_bbox || null}
                image2InitialCenter={sealDetectionResult2?.center || uploadImage2Mutation.data?.seal_center || null}
                onConfirm={handleBatchConfirm}
                onCancel={handleBatchCancel}
              />
            )}
          </DialogContent>
        </Dialog>
      )}
    </Box>
  )
}

export default ComparisonForm

