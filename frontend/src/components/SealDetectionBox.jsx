import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Box, Button, TextField, Typography, Paper, Alert } from '@mui/material'
import { imageAPI } from '../services/api'

function SealDetectionBox({ imageId, initialBbox, initialCenter, onConfirm, onCancel, showCancelButton = true }) {
  const canvasRef = useRef(null)
  const imageRef = useRef(null)
  const [imageLoaded, setImageLoaded] = useState(false)
  const [imageUrl, setImageUrl] = useState(null)
  const [bbox, setBbox] = useState(initialBbox || null)
  const [center, setCenter] = useState(initialCenter || null)
  const [scale, setScale] = useState(1)
  const [isDragging, setIsDragging] = useState(false)
  const [dragType, setDragType] = useState(null) // 'move', 'resize-nw', 'resize-ne', 'resize-sw', 'resize-se', etc.
  const [dragStart, setDragStart] = useState({ x: 0, y: 0, bbox: null })
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 })
  const [hoverType, setHoverType] = useState(null) // 滑鼠懸停時的類型
  const dragStateRef = useRef({ isDragging: false, dragType: null, dragStart: null, scale: 1, imageSize: { width: 0, height: 0 } })

  // 載入圖片
  useEffect(() => {
    if (imageId) {
      const url = imageAPI.getFile(imageId)
      setImageUrl(url)
    }
  }, [imageId])

  // 繪製畫布
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !imageRef.current || !imageLoaded) return

    const ctx = canvas.getContext('2d')
    const img = imageRef.current
    const displayWidth = Math.min(img.width, 800)
    const displayHeight = (img.height / img.width) * displayWidth
    
    canvas.width = displayWidth
    canvas.height = displayHeight
    const currentScale = displayWidth / img.width
    setScale(currentScale)
    setImageSize({ width: displayWidth, height: displayHeight })

    // 清除畫布
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // 繪製圖片
    ctx.drawImage(img, 0, 0, displayWidth, displayHeight)

    // 如果沒有 bbox，不繪製框選
    if (!bbox) return

    // 繪製框選區域
    const scaledBbox = {
      x: bbox.x * currentScale,
      y: bbox.y * currentScale,
      width: bbox.width * currentScale,
      height: bbox.height * currentScale
    }

    // 繪製半透明背景（外部區域較暗）
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    
    // 在框選區域內繪製更淺的半透明遮罩（而不是完全清除）
    ctx.fillStyle = 'rgba(0, 0, 0, 0.1)'
    ctx.fillRect(scaledBbox.x, scaledBbox.y, scaledBbox.width, scaledBbox.height)

    // 繪製邊框
    ctx.strokeStyle = '#2196F3'
    ctx.lineWidth = 2
    ctx.setLineDash([])
    ctx.strokeRect(scaledBbox.x, scaledBbox.y, scaledBbox.width, scaledBbox.height)

    // 繪製控制點
    const handleSize = 8
    const handles = [
      { x: scaledBbox.x, y: scaledBbox.y }, // 左上
      { x: scaledBbox.x + scaledBbox.width / 2, y: scaledBbox.y }, // 上中
      { x: scaledBbox.x + scaledBbox.width, y: scaledBbox.y }, // 右上
      { x: scaledBbox.x + scaledBbox.width, y: scaledBbox.y + scaledBbox.height / 2 }, // 右中
      { x: scaledBbox.x + scaledBbox.width, y: scaledBbox.y + scaledBbox.height }, // 右下
      { x: scaledBbox.x + scaledBbox.width / 2, y: scaledBbox.y + scaledBbox.height }, // 下中
      { x: scaledBbox.x, y: scaledBbox.y + scaledBbox.height }, // 左下
      { x: scaledBbox.x, y: scaledBbox.y + scaledBbox.height / 2 }, // 左中
    ]

    ctx.fillStyle = '#2196F3'
    handles.forEach(handle => {
      ctx.fillRect(handle.x - handleSize / 2, handle.y - handleSize / 2, handleSize, handleSize)
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 1
      ctx.strokeRect(handle.x - handleSize / 2, handle.y - handleSize / 2, handleSize, handleSize)
    })
  }, [bbox, imageLoaded])

  useEffect(() => {
    drawCanvas()
  }, [drawCanvas])

  // 獲取控制點類型
  const getHandleType = (x, y) => {
    if (!bbox) return null
    
    const scaledBbox = {
      x: bbox.x * scale,
      y: bbox.y * scale,
      width: bbox.width * scale,
      height: bbox.height * scale
    }
    const handleSize = 12
    const handles = [
      { x: scaledBbox.x, y: scaledBbox.y, type: 'resize-nw' },
      { x: scaledBbox.x + scaledBbox.width / 2, y: scaledBbox.y, type: 'resize-n' },
      { x: scaledBbox.x + scaledBbox.width, y: scaledBbox.y, type: 'resize-ne' },
      { x: scaledBbox.x + scaledBbox.width, y: scaledBbox.y + scaledBbox.height / 2, type: 'resize-e' },
      { x: scaledBbox.x + scaledBbox.width, y: scaledBbox.y + scaledBbox.height, type: 'resize-se' },
      { x: scaledBbox.x + scaledBbox.width / 2, y: scaledBbox.y + scaledBbox.height, type: 'resize-s' },
      { x: scaledBbox.x, y: scaledBbox.y + scaledBbox.height, type: 'resize-sw' },
      { x: scaledBbox.x, y: scaledBbox.y + scaledBbox.height / 2, type: 'resize-w' },
    ]

    for (const handle of handles) {
      const dx = x - handle.x
      const dy = y - handle.y
      if (Math.abs(dx) < handleSize && Math.abs(dy) < handleSize) {
        return handle.type
      }
    }

    // 檢查是否在框內（移動）
    if (
      x >= scaledBbox.x &&
      x <= scaledBbox.x + scaledBbox.width &&
      y >= scaledBbox.y &&
      y <= scaledBbox.y + scaledBbox.height
    ) {
      return 'move'
    }

    return null
  }

  // 處理滑鼠按下
  const handleMouseDown = (e) => {
    const canvas = canvasRef.current
    if (!canvas || !bbox) return

    e.preventDefault() // 防止文字選擇

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    const type = getHandleType(x, y)
    if (type) {
      setIsDragging(true)
      setDragType(type)
      setDragStart({ x, y, bbox: { ...bbox } })
      
      // 添加全局事件監聽器，確保即使滑鼠移出 canvas 也能繼續拖曳
      document.addEventListener('mousemove', handleGlobalMouseMove)
      document.addEventListener('mouseup', handleGlobalMouseUp)
    }
  }

  // 更新 dragStateRef
  useEffect(() => {
    dragStateRef.current = { isDragging, dragType, dragStart, scale, imageSize }
  }, [isDragging, dragType, dragStart, scale, imageSize])

  // 全局滑鼠移動處理（用於拖曳時滑鼠移出 canvas 的情況）
  const handleGlobalMouseMove = useCallback((e) => {
    const state = dragStateRef.current
    if (!state.isDragging || !state.dragType || !state.dragStart?.bbox) return

    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    const xScaled = x / state.scale
    const yScaled = y / state.scale
    const startX = state.dragStart.x / state.scale
    const startY = state.dragStart.y / state.scale

    const dx = xScaled - startX
    const dy = yScaled - startY

    let newBbox = { ...state.dragStart.bbox }

    if (state.dragType === 'move') {
      newBbox.x = Math.max(0, Math.min(state.dragStart.bbox.x + dx, state.imageSize.width / state.scale - state.dragStart.bbox.width))
      newBbox.y = Math.max(0, Math.min(state.dragStart.bbox.y + dy, state.imageSize.height / state.scale - state.dragStart.bbox.height))
    } else if (state.dragType.startsWith('resize-')) {
      const direction = state.dragType.split('-')[1]
      
      if (direction.includes('n')) {
        const newY = Math.max(0, state.dragStart.bbox.y + dy)
        const newHeight = state.dragStart.bbox.height - (newY - state.dragStart.bbox.y)
        if (newHeight >= 10) {
          newBbox.y = newY
          newBbox.height = newHeight
        }
      }
      if (direction.includes('s')) {
        newBbox.height = Math.max(10, state.dragStart.bbox.height + dy)
      }
      if (direction.includes('w')) {
        const newX = Math.max(0, state.dragStart.bbox.x + dx)
        const newWidth = state.dragStart.bbox.width - (newX - state.dragStart.bbox.x)
        if (newWidth >= 10) {
          newBbox.x = newX
          newBbox.width = newWidth
        }
      }
      if (direction.includes('e')) {
        newBbox.width = Math.max(10, state.dragStart.bbox.width + dx)
      }
    }

    // 確保不超出邊界
    newBbox.x = Math.max(0, Math.min(newBbox.x, state.imageSize.width / state.scale - newBbox.width))
    newBbox.y = Math.max(0, Math.min(newBbox.y, state.imageSize.height / state.scale - newBbox.height))
    newBbox.width = Math.min(newBbox.width, state.imageSize.width / state.scale - newBbox.x)
    newBbox.height = Math.min(newBbox.height, state.imageSize.height / state.scale - newBbox.y)

    setBbox(newBbox)
    updateCenter(newBbox)
  }, [])

  // 全局滑鼠釋放處理
  const handleGlobalMouseUp = useCallback(() => {
    const state = dragStateRef.current
    if (state.isDragging) {
      setIsDragging(false)
      setDragType(null)
      document.removeEventListener('mousemove', handleGlobalMouseMove)
      document.removeEventListener('mouseup', handleGlobalMouseUp)
    }
  }, [handleGlobalMouseMove])

  // 處理滑鼠移動（僅用於更新游標樣式，拖曳由全局事件處理）
  const handleMouseMove = (e) => {
    const canvas = canvasRef.current
    if (!canvas || !bbox || isDragging) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    // 更新游標樣式
    const type = getHandleType(x, y)
    setHoverType(type)
  }

  // 處理滑鼠釋放（canvas 內）
  const handleMouseUp = () => {
    handleGlobalMouseUp()
  }

  // 清理事件監聽器
  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', handleGlobalMouseMove)
      document.removeEventListener('mouseup', handleGlobalMouseUp)
    }
  }, [handleGlobalMouseMove, handleGlobalMouseUp])

  // 獲取游標樣式
  const getCursorStyle = () => {
    if (isDragging) {
      if (dragType === 'move') return 'grabbing'
      if (dragType?.startsWith('resize-')) {
        const direction = dragType.split('-')[1]
        if (direction === 'nw' || direction === 'se') return 'nwse-resize'
        if (direction === 'ne' || direction === 'sw') return 'nesw-resize'
        if (direction === 'n' || direction === 's') return 'ns-resize'
        if (direction === 'e' || direction === 'w') return 'ew-resize'
      }
      return 'grabbing'
    }
    
    if (hoverType) {
      if (hoverType === 'move') return 'grab'
      if (hoverType?.startsWith('resize-')) {
        const direction = hoverType.split('-')[1]
        if (direction === 'nw' || direction === 'se') return 'nwse-resize'
        if (direction === 'ne' || direction === 'sw') return 'nesw-resize'
        if (direction === 'n' || direction === 's') return 'ns-resize'
        if (direction === 'e' || direction === 'w') return 'ew-resize'
      }
    }
    
    return 'default'
  }

  // 更新中心點
  const updateCenter = (newBbox) => {
    setCenter({
      center_x: newBbox.x + newBbox.width / 2,
      center_y: newBbox.y + newBbox.height / 2,
      radius: Math.sqrt(newBbox.width ** 2 + newBbox.height ** 2) / 2
    })
  }

  // 處理數值輸入
  const handleBboxChange = (field, value) => {
    const numValue = parseInt(value) || 0
    const newBbox = { ...bbox, [field]: Math.max(0, numValue) }
    
    // 確保不超出邊界
    if (imageSize.width > 0 && imageSize.height > 0) {
      newBbox.x = Math.min(newBbox.x, imageSize.width / scale - newBbox.width)
      newBbox.y = Math.min(newBbox.y, imageSize.height / scale - newBbox.height)
      newBbox.width = Math.min(newBbox.width, imageSize.width / scale - newBbox.x)
      newBbox.height = Math.min(newBbox.height, imageSize.height / scale - newBbox.y)
    }
    
    setBbox(newBbox)
    updateCenter(newBbox)
  }

  // 處理確認
  const handleConfirm = () => {
    if (onConfirm && bbox) {
      // 確保 bbox 的值是整數
      const normalizedBbox = {
        x: Math.round(bbox.x),
        y: Math.round(bbox.y),
        width: Math.round(bbox.width),
        height: Math.round(bbox.height)
      }
      
      // 計算或使用現有的 center
      const normalizedCenter = center || {
        center_x: Math.round(normalizedBbox.x + normalizedBbox.width / 2),
        center_y: Math.round(normalizedBbox.y + normalizedBbox.height / 2),
        radius: Math.sqrt(normalizedBbox.width ** 2 + normalizedBbox.height ** 2) / 2
      }
      
      // 確保 center 的值是正確的類型
      const finalCenter = {
        center_x: Math.round(normalizedCenter.center_x),
        center_y: Math.round(normalizedCenter.center_y),
        radius: typeof normalizedCenter.radius === 'number' ? normalizedCenter.radius : Math.sqrt(normalizedBbox.width ** 2 + normalizedBbox.height ** 2) / 2
      }
      
      onConfirm({ 
        bbox: normalizedBbox, 
        center: finalCenter
      })
    }
  }

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        調整印鑑位置
      </Typography>
      
      <Box sx={{ mb: 2 }}>
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={(e) => {
            handleMouseUp()
            setHoverType(null)
          }}
          style={{
            cursor: getCursorStyle(),
            border: '1px solid #ddd',
            maxWidth: '100%',
            height: 'auto',
            userSelect: 'none',
            WebkitUserSelect: 'none',
            MozUserSelect: 'none',
            msUserSelect: 'none'
          }}
        />
        <img
          ref={imageRef}
          src={imageUrl}
          alt="預覽"
          style={{ display: 'none' }}
          onLoad={() => {
            setImageLoaded(true)
            if (initialBbox) {
              setBbox(initialBbox)
              if (initialCenter) {
                setCenter(initialCenter)
              } else {
                updateCenter(initialBbox)
              }
            } else {
              // 如果沒有初始 bbox，設置默認值（圖像中心區域，較小的默認框）
              const img = imageRef.current
              if (img) {
                const defaultBbox = {
                  x: Math.round(img.width * 0.35),
                  y: Math.round(img.height * 0.35),
                  width: Math.round(img.width * 0.3),
                  height: Math.round(img.height * 0.3)
                }
                setBbox(defaultBbox)
                updateCenter(defaultBbox)
              }
            }
          }}
        />
      </Box>

      {bbox && (
        <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2, mb: 2 }}>
          <TextField
            label="X 位置"
            type="number"
            value={Math.round(bbox.x)}
            onChange={(e) => handleBboxChange('x', e.target.value)}
            size="small"
          />
          <TextField
            label="Y 位置"
            type="number"
            value={Math.round(bbox.y)}
            onChange={(e) => handleBboxChange('y', e.target.value)}
            size="small"
          />
          <TextField
            label="寬度"
            type="number"
            value={Math.round(bbox.width)}
            onChange={(e) => handleBboxChange('width', e.target.value)}
            size="small"
          />
          <TextField
            label="高度"
            type="number"
            value={Math.round(bbox.height)}
            onChange={(e) => handleBboxChange('height', e.target.value)}
            size="small"
          />
        </Box>
      )}

      <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
        {showCancelButton && onCancel && (
          <Button variant="outlined" onClick={onCancel}>
            取消
          </Button>
        )}
        <Button variant="contained" onClick={handleConfirm}>
          確認
        </Button>
      </Box>
    </Paper>
  )
}

export default SealDetectionBox

