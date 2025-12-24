import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Box, Paper, Typography, IconButton, Chip } from '@mui/material'
import { Edit as EditIcon, CheckCircle as CheckCircleIcon } from '@mui/icons-material'
import { imageAPI } from '../services/api'

function ImagePreview({ image, label, onEdit, showSealIndicator = true, onPreview }) {
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 })
  const imgRef = useRef(null)
  const containerRef = useRef(null)
  const canvasRef = useRef(null)


  // 繪製框選框到 canvas
  const drawBbox = useCallback(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    
    if (!canvas || !img || !image?.seal_bbox || !img.complete) {
      return
    }

    const ctx = canvas.getContext('2d')
    const container = containerRef.current
    if (!container) return

    // 獲取容器的實際顯示位置和尺寸
    const containerRect = container.getBoundingClientRect()
    
    const containerWidth = containerRect.width
    const containerHeight = containerRect.height

    canvas.width = containerWidth
    canvas.height = containerHeight
    
    // 計算圖片內容的實際顯示尺寸（考慮 objectFit: contain）
    // 當使用 objectFit: contain 時，圖片會保持寬高比，實際顯示尺寸可能小於容器尺寸
    const imgAspect = img.naturalWidth / img.naturalHeight
    const containerAspect = containerWidth / containerHeight
    
    let imgDisplayWidth, imgDisplayHeight, imgOffsetX, imgOffsetY
    
    if (imgAspect > containerAspect) {
      // 圖片較寬，以容器寬度為準
      imgDisplayWidth = containerWidth
      imgDisplayHeight = containerWidth / imgAspect
      imgOffsetX = 0
      imgOffsetY = (containerHeight - imgDisplayHeight) / 2
    } else {
      // 圖片較高，以容器高度為準
      imgDisplayHeight = containerHeight
      imgDisplayWidth = containerHeight * imgAspect
      imgOffsetX = (containerWidth - imgDisplayWidth) / 2
      imgOffsetY = 0
    }
    
    // 計算縮放比例：從圖片原始尺寸縮放到顯示尺寸
    // bbox 坐標是相對於圖片原始尺寸（naturalWidth/naturalHeight）的
    // 參考 SealDetectionBox: currentScale = displayWidth / img.width
    // 這裡 imgDisplayWidth 對應 displayWidth，img.naturalWidth 對應 img.width（當圖片隱藏時）
    const currentScale = imgDisplayWidth / img.naturalWidth

    // 清除畫布
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // 如果沒有 bbox，不繪製
    if (!image.seal_bbox) return

    // 驗證 bbox 格式
    const bbox = image.seal_bbox
    if (bbox.x === undefined || bbox.y === undefined || bbox.width === undefined || bbox.height === undefined) {
      console.warn('ImagePreview: bbox 格式不正確:', bbox)
      return
    }

    // 計算 bbox 在顯示尺寸中的位置
    // bbox 坐標是相對於圖片原始尺寸的，需要轉換到顯示坐標系統
    // 參考 SealDetectionBox: scaledBbox.x = bbox.x * currentScale
    // 但這裡需要加上圖片在容器中的偏移
    const scaledBbox = {
      x: imgOffsetX + bbox.x * currentScale,
      y: imgOffsetY + bbox.y * currentScale,
      width: bbox.width * currentScale,
      height: bbox.height * currentScale
    }

    // 繪製半透明背景（外部區域較暗）
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    
    // 在框選區域內繪製更淺的半透明遮罩
    ctx.fillStyle = 'rgba(0, 0, 0, 0.1)'
    ctx.fillRect(scaledBbox.x, scaledBbox.y, scaledBbox.width, scaledBbox.height)

    // 繪製邊框
    ctx.strokeStyle = '#2196F3'
    ctx.lineWidth = 2
    ctx.setLineDash([])
    ctx.strokeRect(scaledBbox.x, scaledBbox.y, scaledBbox.width, scaledBbox.height)
  }, [image])

  // 當圖像或 bbox 變化時重新繪製
  useEffect(() => {
    if (imgRef.current?.complete) {
      drawBbox()
    }
  }, [image, drawBbox])

  // 當容器大小變化時重新繪製
  useEffect(() => {
    const handleResize = () => {
      if (imgRef.current?.complete) {
        drawBbox()
      }
    }
    
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [drawBbox])

  if (!image) {
    return (
      <Paper
        sx={{
          p: 2,
          textAlign: 'center',
          minHeight: 200,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          border: '2px dashed #ddd',
          backgroundColor: '#fafafa',
        }}
      >
        <Typography variant="body2" color="text.secondary">
          {label || '未上傳圖像'}
        </Typography>
      </Paper>
    )
  }

  const imageUrl = imageAPI.getFile(image.id)
  const hasSeal = image.seal_bbox !== null && image.seal_bbox !== undefined

  return (
    <Paper
      sx={{
        p: 2,
        position: 'relative',
        '&:hover': {
          boxShadow: 3,
        },
      }}
    >
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="subtitle2" fontWeight="bold">
          {label}
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
          {showSealIndicator && (
            <Chip
              icon={hasSeal ? <CheckCircleIcon /> : null}
              label={hasSeal ? '已標記' : '未標記'}
              size="small"
              color={hasSeal ? 'success' : 'default'}
              variant={hasSeal ? 'filled' : 'outlined'}
            />
          )}
          {onEdit && (
            <IconButton
              size="small"
              onClick={onEdit}
              sx={{ ml: 1 }}
              title="編輯印鑑位置"
            >
              <EditIcon fontSize="small" />
            </IconButton>
          )}
        </Box>
      </Box>
      
      <Box
        ref={containerRef}
        onClick={onPreview ? () => onPreview(image) : undefined}
        sx={{
          position: 'relative',
          width: '100%',
          paddingTop: '75%', // 4:3 比例
          backgroundColor: '#f5f5f5',
          borderRadius: 1,
          overflow: 'hidden',
          border: hasSeal ? '2px solid #4caf50' : '1px solid #ddd',
          cursor: onPreview ? 'pointer' : 'default',
          '&:hover': onPreview ? {
            opacity: 0.9,
          } : {},
        }}
      >
        <img
          ref={imgRef}
          src={imageUrl}
          alt={image.filename || '預覽'}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            objectFit: 'contain',
          }}
          onLoad={(e) => {
            setImageSize({
              width: e.target.naturalWidth,
              height: e.target.naturalHeight,
            })
            drawBbox()
          }}
        />
        {hasSeal && (
          <canvas
            ref={canvasRef}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              pointerEvents: 'none', // 不阻止點擊事件
            }}
          />
        )}
      </Box>
      
      {image.filename && (
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{
            mt: 1,
            display: 'block',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {image.filename}
        </Typography>
      )}
    </Paper>
  )
}

export default ImagePreview

