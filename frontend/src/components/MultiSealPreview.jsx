import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Box, Paper, Typography, Chip } from '@mui/material'
import { imageAPI } from '../services/api'

function MultiSealPreview({ image, seals = [], label, onSealClick, onPreview }) {
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 })
  const imgRef = useRef(null)
  const containerRef = useRef(null)
  const canvasRef = useRef(null)

  // 不同印鑑使用不同顏色
  const sealColors = [
    '#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0',
    '#00BCD4', '#FFEB3B', '#795548', '#607D8B', '#E91E63'
  ]

  // 優先使用 image.multiple_seals，否則使用 seals prop
  // 確保數據格式正確（數組）
  const displaySeals = React.useMemo(() => {
    if (image?.multiple_seals && Array.isArray(image.multiple_seals)) {
      return image.multiple_seals
    }
    if (Array.isArray(seals)) {
      return seals
    }
    return []
  }, [image?.multiple_seals, seals])

  // 繪製多個印鑑框選到 canvas
  const drawBboxes = useCallback(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    
    if (!canvas || !img || !displaySeals || displaySeals.length === 0 || !img.complete) {
      return
    }

    const ctx = canvas.getContext('2d')
    const container = containerRef.current
    if (!container) return

    // 獲取容器和圖片的實際顯示位置和尺寸
    const containerRect = container.getBoundingClientRect()
    const imgRect = img.getBoundingClientRect()
    
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

    // 繪製每個印鑑框選
    displaySeals.forEach((seal, index) => {
      // 驗證 seal 數據格式
      if (!seal || !seal.bbox) return
      
      // 驗證 bbox 格式
      const bbox = seal.bbox
      if (bbox.x === undefined || bbox.y === undefined || bbox.width === undefined || bbox.height === undefined) {
        console.warn(`印鑑 ${index + 1} 的 bbox 格式不正確:`, bbox)
        return
      }

      const color = sealColors[index % sealColors.length]
      
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
      if (index === 0) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
      }
      
      // 在框選區域內繪製更淺的半透明遮罩
      ctx.fillStyle = 'rgba(0, 0, 0, 0.1)'
      ctx.fillRect(scaledBbox.x, scaledBbox.y, scaledBbox.width, scaledBbox.height)

      // 繪製邊框
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.setLineDash([])
      ctx.strokeRect(scaledBbox.x, scaledBbox.y, scaledBbox.width, scaledBbox.height)

      // 繪製編號標籤
      ctx.fillStyle = color
      ctx.font = 'bold 14px Arial'
      ctx.fillText(
        `${index + 1}`,
        scaledBbox.x + 5,
        scaledBbox.y + 18
      )
    })
  }, [displaySeals, sealColors])

  // 當圖像或印鑑列表變化時重新繪製
  useEffect(() => {
    if (imgRef.current?.complete) {
      drawBboxes()
    }
  }, [image, image?.multiple_seals, seals, drawBboxes])

  // 當容器大小變化時重新繪製
  useEffect(() => {
    const handleResize = () => {
      if (imgRef.current?.complete) {
        drawBboxes()
      }
    }
    
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [drawBboxes])

  // 處理點擊事件
  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img || !img.complete) return

    const canvasRect = canvas.getBoundingClientRect()
    const x = e.clientX - canvasRect.left
    const y = e.clientY - canvasRect.top

    const container = containerRef.current
    if (!container) return

    // 獲取容器和圖片的實際顯示位置和尺寸
    const containerRect = container.getBoundingClientRect()
    const imgRect = img.getBoundingClientRect()
    
    // 圖片實際顯示尺寸（考慮 objectFit: contain）
    const imgDisplayWidth = imgRect.width
    
    // 圖片在容器中的偏移（相對於容器左上角）
    const imgOffsetX = imgRect.left - containerRect.left
    const imgOffsetY = imgRect.top - containerRect.top

    // 計算縮放比例：從圖片原始尺寸縮放到顯示尺寸
    const currentScale = imgDisplayWidth / img.naturalWidth

    // 檢查點擊是否在某個印鑑框內
    let clickedOnSeal = false
    if (displaySeals && displaySeals.length > 0 && onSealClick) {
      for (let i = displaySeals.length - 1; i >= 0; i--) {
        const seal = displaySeals[i]
        if (!seal || !seal.bbox) continue

        const bbox = seal.bbox
        // 驗證 bbox 格式
        if (bbox.x === undefined || bbox.y === undefined || bbox.width === undefined || bbox.height === undefined) {
          continue
        }
        // 使用與 drawBboxes 相同的計算邏輯
        const scaledBbox = {
          x: imgOffsetX + bbox.x * currentScale,
          y: imgOffsetY + bbox.y * currentScale,
          width: bbox.width * currentScale,
          height: bbox.height * currentScale
        }

        if (
          x >= scaledBbox.x &&
          x <= scaledBbox.x + scaledBbox.width &&
          y >= scaledBbox.y &&
          y <= scaledBbox.y + scaledBbox.height
        ) {
          onSealClick(i, seal)
          clickedOnSeal = true
          break
        }
      }
    }

    // 如果沒有點擊到印鑑框，且提供了 onPreview，則觸發預覽
    if (!clickedOnSeal && onPreview && image) {
      onPreview(image)
    }
  }

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
  // 使用 displaySeals 判斷是否有印鑑
  const hasSeals = displaySeals && displaySeals.length > 0

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
          {hasSeals && (
            <Chip
              label={`已檢測 ${displaySeals.length} 個印鑑`}
              size="small"
              color="success"
              variant="filled"
            />
          )}
        </Box>
      </Box>
      
      <Box
        ref={containerRef}
        sx={{
          position: 'relative',
          width: '100%',
          paddingTop: '75%', // 4:3 比例
          backgroundColor: '#f5f5f5',
          borderRadius: 1,
          overflow: 'hidden',
          border: hasSeals ? '2px solid #4caf50' : '1px solid #ddd',
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
            drawBboxes()
          }}
        />
        {hasSeals && (
          <canvas
            ref={canvasRef}
            onClick={handleCanvasClick}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              cursor: (onSealClick || onPreview) ? 'pointer' : 'default',
            }}
          />
        )}
        {!hasSeals && onPreview && (
          <Box
            onClick={() => onPreview(image)}
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              cursor: 'pointer',
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

export default MultiSealPreview

