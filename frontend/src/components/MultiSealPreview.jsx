import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Box, Paper, Typography, Chip } from '@mui/material'
import { imageAPI } from '../services/api'

function MultiSealPreview({ image, seals = [], label, onSealClick }) {
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 })
  const imgRef = useRef(null)
  const containerRef = useRef(null)
  const canvasRef = useRef(null)

  // 不同印鑑使用不同顏色
  const sealColors = [
    '#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0',
    '#00BCD4', '#FFEB3B', '#795548', '#607D8B', '#E91E63'
  ]

  // 繪製多個印鑑框選到 canvas
  const drawBboxes = useCallback(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    
    if (!canvas || !img || !seals || seals.length === 0) {
      return
    }

    const ctx = canvas.getContext('2d')
    const container = containerRef.current
    if (!container) return

    // 計算顯示尺寸和縮放比例
    const containerWidth = container.offsetWidth
    const containerHeight = container.offsetHeight
    
    // 計算圖像在容器中的顯示尺寸（保持寬高比）
    const imgAspect = img.naturalWidth / img.naturalHeight
    const containerAspect = containerWidth / containerHeight
    
    let displayWidth, displayHeight
    if (imgAspect > containerAspect) {
      displayWidth = containerWidth
      displayHeight = containerWidth / imgAspect
    } else {
      displayHeight = containerHeight
      displayWidth = containerHeight * imgAspect
    }

    canvas.width = containerWidth
    canvas.height = containerHeight
    
    // 計算縮放比例
    const currentScale = displayWidth / img.naturalWidth

    // 清除畫布
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // 計算偏移量
    const offsetX = (containerWidth - displayWidth) / 2
    const offsetY = (containerHeight - displayHeight) / 2

    // 繪製每個印鑑框選
    seals.forEach((seal, index) => {
      if (!seal.bbox) return

      const bbox = seal.bbox
      const color = sealColors[index % sealColors.length]
      
      const scaledBbox = {
        x: offsetX + bbox.x * currentScale,
        y: offsetY + bbox.y * currentScale,
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
  }, [seals, sealColors])

  // 當圖像或印鑑列表變化時重新繪製
  useEffect(() => {
    if (imgRef.current?.complete) {
      drawBboxes()
    }
  }, [image, seals, drawBboxes])

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
    if (!seals || seals.length === 0 || !onSealClick) return

    const canvas = canvasRef.current
    const img = imgRef.current
    if (!canvas || !img) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    const container = containerRef.current
    if (!container) return

    const containerWidth = container.offsetWidth
    const containerHeight = container.offsetHeight
    
    const imgAspect = img.naturalWidth / img.naturalHeight
    const containerAspect = containerWidth / containerHeight
    
    let displayWidth, displayHeight
    if (imgAspect > containerAspect) {
      displayWidth = containerWidth
      displayHeight = containerWidth / imgAspect
    } else {
      displayHeight = containerHeight
      displayWidth = containerHeight * imgAspect
    }

    const currentScale = displayWidth / img.naturalWidth
    const offsetX = (containerWidth - displayWidth) / 2
    const offsetY = (containerHeight - displayHeight) / 2

    // 檢查點擊是否在某個印鑑框內
    for (let i = seals.length - 1; i >= 0; i--) {
      const seal = seals[i]
      if (!seal.bbox) continue

      const bbox = seal.bbox
      const scaledBbox = {
        x: offsetX + bbox.x * currentScale,
        y: offsetY + bbox.y * currentScale,
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
        break
      }
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
  const hasSeals = seals && seals.length > 0

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
              label={`已檢測 ${seals.length} 個印鑑`}
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
              cursor: onSealClick ? 'pointer' : 'default',
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

