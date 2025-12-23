import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Box, Button, TextField, Typography, Paper, Alert, IconButton, Chip } from '@mui/material'
import { Delete as DeleteIcon, Add as AddIcon } from '@mui/icons-material'
import { imageAPI } from '../services/api'

function MultiSealDetectionBox({ imageId, initialSeals = [], onConfirm, onCancel, showCancelButton = true }) {
  const canvasRef = useRef(null)
  const imageRef = useRef(null)
  const [imageLoaded, setImageLoaded] = useState(false)
  const [imageUrl, setImageUrl] = useState(null)
  const [seals, setSeals] = useState(initialSeals || [])
  const [selectedSealIndex, setSelectedSealIndex] = useState(null)
  const [scale, setScale] = useState(1)
  const [isDragging, setIsDragging] = useState(false)
  const [dragType, setDragType] = useState(null)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0, sealIndex: null, bbox: null })
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 })
  const dragStateRef = useRef({ isDragging: false, dragType: null, dragStart: null, scale: 1, imageSize: { width: 0, height: 0 } })

  // 不同印鑑使用不同顏色
  const sealColors = [
    '#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0',
    '#00BCD4', '#FFEB3B', '#795548', '#607D8B', '#E91E63'
  ]

  // 載入圖片
  useEffect(() => {
    if (imageId) {
      const url = imageAPI.getFile(imageId)
      setImageUrl(url)
    }
  }, [imageId])

  // 初始化 seals
  useEffect(() => {
    if (initialSeals && initialSeals.length > 0) {
      setSeals(initialSeals)
    }
  }, [initialSeals])

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

    // 繪製所有印鑑框選
    seals.forEach((seal, index) => {
      if (!seal.bbox) return

      const color = sealColors[index % sealColors.length]
      const isSelected = selectedSealIndex === index
      
      const scaledBbox = {
        x: seal.bbox.x * currentScale,
        y: seal.bbox.y * currentScale,
        width: seal.bbox.width * currentScale,
        height: seal.bbox.height * currentScale
      }

      // 繪製半透明背景（外部區域較暗）
      if (index === 0) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
      }
      
      // 在框選區域內繪製更淺的半透明遮罩
      ctx.fillStyle = 'rgba(0, 0, 0, 0.1)'
      ctx.fillRect(scaledBbox.x, scaledBbox.y, scaledBbox.width, scaledBbox.height)

      // 繪製邊框（選中的更粗）
      ctx.strokeStyle = color
      ctx.lineWidth = isSelected ? 3 : 2
      ctx.setLineDash([])
      ctx.strokeRect(scaledBbox.x, scaledBbox.y, scaledBbox.width, scaledBbox.height)

      // 繪製控制點（僅選中的印鑑）
      if (isSelected) {
        const handleSize = 8
        const handles = [
          { x: scaledBbox.x, y: scaledBbox.y },
          { x: scaledBbox.x + scaledBbox.width / 2, y: scaledBbox.y },
          { x: scaledBbox.x + scaledBbox.width, y: scaledBbox.y },
          { x: scaledBbox.x + scaledBbox.width, y: scaledBbox.y + scaledBbox.height / 2 },
          { x: scaledBbox.x + scaledBbox.width, y: scaledBbox.y + scaledBbox.height },
          { x: scaledBbox.x + scaledBbox.width / 2, y: scaledBbox.y + scaledBbox.height },
          { x: scaledBbox.x, y: scaledBbox.y + scaledBbox.height },
          { x: scaledBbox.x, y: scaledBbox.y + scaledBbox.height / 2 },
        ]

        ctx.fillStyle = color
        handles.forEach(handle => {
          ctx.fillRect(handle.x - handleSize / 2, handle.y - handleSize / 2, handleSize, handleSize)
          ctx.strokeStyle = '#fff'
          ctx.lineWidth = 1
          ctx.strokeRect(handle.x - handleSize / 2, handle.y - handleSize / 2, handleSize, handleSize)
        })
      }

      // 繪製編號標籤
      ctx.fillStyle = color
      ctx.font = 'bold 16px Arial'
      ctx.fillText(
        `${index + 1}`,
        scaledBbox.x + 5,
        scaledBbox.y + 20
      )
    })
  }, [seals, selectedSealIndex, imageLoaded, sealColors])

  useEffect(() => {
    drawCanvas()
  }, [drawCanvas])

  // 獲取控制點類型
  const getHandleType = (x, y, sealIndex) => {
    if (sealIndex === null || !seals[sealIndex]?.bbox) return null
    
    const seal = seals[sealIndex]
    const scaledBbox = {
      x: seal.bbox.x * scale,
      y: seal.bbox.y * scale,
      width: seal.bbox.width * scale,
      height: seal.bbox.height * scale
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
    if (!canvas) return

    e.preventDefault()

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    // 檢查點擊是否在某個印鑑上
    let clickedSealIndex = null
    for (let i = seals.length - 1; i >= 0; i--) {
      if (!seals[i]?.bbox) continue
      const type = getHandleType(x, y, i)
      if (type) {
        clickedSealIndex = i
        setSelectedSealIndex(i)
        setIsDragging(true)
        setDragType(type)
        setDragStart({ x, y, sealIndex: i, bbox: { ...seals[i].bbox } })
        
        document.addEventListener('mousemove', handleGlobalMouseMove)
        document.addEventListener('mouseup', handleGlobalMouseUp)
        break
      }
    }

    // 如果沒有點擊到任何印鑑，取消選擇
    if (clickedSealIndex === null) {
      setSelectedSealIndex(null)
    }
  }

  // 更新 dragStateRef
  useEffect(() => {
    dragStateRef.current = { isDragging, dragType, dragStart, scale, imageSize, seals }
  }, [isDragging, dragType, dragStart, scale, imageSize, seals])

  // 全局滑鼠移動處理
  const handleGlobalMouseMove = useCallback((e) => {
    const state = dragStateRef.current
    if (!state.isDragging || !state.dragType || state.dragStart?.sealIndex === null || !state.seals) return

    const canvas = canvasRef.current
    if (!canvas) return

    const rect = canvas.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    const sealIndex = state.dragStart.sealIndex
    const seal = state.seals[sealIndex]
    if (!seal?.bbox) return

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

    // 更新對應的印鑑
    const newSeals = [...state.seals]
    newSeals[sealIndex] = {
      ...seal,
      bbox: newBbox,
      center: {
        center_x: Math.round(newBbox.x + newBbox.width / 2),
        center_y: Math.round(newBbox.y + newBbox.height / 2),
        radius: Math.sqrt(newBbox.width ** 2 + newBbox.height ** 2) / 2
      }
    }
    setSeals(newSeals)
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

  // 清理事件監聽器
  useEffect(() => {
    return () => {
      document.removeEventListener('mousemove', handleGlobalMouseMove)
      document.removeEventListener('mouseup', handleGlobalMouseUp)
    }
  }, [handleGlobalMouseMove, handleGlobalMouseUp])

  // 添加新印鑑
  const handleAddSeal = () => {
    if (!imageRef.current) return
    
    const img = imageRef.current
    const defaultBbox = {
      x: Math.round(img.width * 0.35),
      y: Math.round(img.height * 0.35),
      width: Math.round(img.width * 0.3),
      height: Math.round(img.height * 0.3)
    }
    
    const newSeal = {
      bbox: defaultBbox,
      center: {
        center_x: Math.round(defaultBbox.x + defaultBbox.width / 2),
        center_y: Math.round(defaultBbox.y + defaultBbox.height / 2),
        radius: Math.sqrt(defaultBbox.width ** 2 + defaultBbox.height ** 2) / 2
      },
      confidence: 0.5
    }
    
    setSeals([...seals, newSeal])
    setSelectedSealIndex(seals.length)
  }

  // 刪除選中的印鑑
  const handleDeleteSeal = () => {
    if (selectedSealIndex === null) return
    
    const newSeals = seals.filter((_, index) => index !== selectedSealIndex)
    setSeals(newSeals)
    setSelectedSealIndex(null)
  }

  // 處理確認
  const handleConfirm = () => {
    if (onConfirm && seals.length > 0) {
      const normalizedSeals = seals.map(seal => ({
        bbox: {
          x: Math.round(seal.bbox.x),
          y: Math.round(seal.bbox.y),
          width: Math.round(seal.bbox.width),
          height: Math.round(seal.bbox.height)
        },
        center: {
          center_x: Math.round(seal.center.center_x),
          center_y: Math.round(seal.center.center_y),
          radius: typeof seal.center.radius === 'number' ? seal.center.radius : Math.sqrt(seal.bbox.width ** 2 + seal.bbox.height ** 2) / 2
        },
        confidence: seal.confidence || 0.5
      }))
      
      onConfirm(normalizedSeals)
    }
  }

  return (
    <Paper sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          調整多個印鑑位置
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            size="small"
            startIcon={<AddIcon />}
            onClick={handleAddSeal}
          >
            添加
          </Button>
          {selectedSealIndex !== null && (
            <Button
              variant="outlined"
              color="error"
              size="small"
              startIcon={<DeleteIcon />}
              onClick={handleDeleteSeal}
            >
              刪除
            </Button>
          )}
        </Box>
      </Box>
      
      <Box sx={{ mb: 2 }}>
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          style={{
            cursor: isDragging ? 'grabbing' : 'default',
            border: '1px solid #ddd',
            maxWidth: '100%',
            height: 'auto',
            userSelect: 'none',
          }}
        />
        <img
          ref={imageRef}
          src={imageUrl}
          alt="預覽"
          style={{ display: 'none' }}
          onLoad={() => {
            setImageLoaded(true)
            if (seals.length === 0 && initialSeals && initialSeals.length > 0) {
              setSeals(initialSeals)
            }
          }}
        />
      </Box>

      {seals.length > 0 && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            檢測到的印鑑 ({seals.length} 個)
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
            {seals.map((seal, index) => (
              <Chip
                key={index}
                label={`印鑑 ${index + 1}`}
                color={selectedSealIndex === index ? 'primary' : 'default'}
                onClick={() => setSelectedSealIndex(index)}
                sx={{
                  backgroundColor: selectedSealIndex === index ? sealColors[index % sealColors.length] : undefined,
                  color: selectedSealIndex === index ? '#fff' : undefined,
                }}
              />
            ))}
          </Box>
        </Box>
      )}

      <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
        {showCancelButton && onCancel && (
          <Button variant="outlined" onClick={onCancel}>
            取消
          </Button>
        )}
        <Button 
          variant="contained" 
          onClick={handleConfirm}
          disabled={seals.length === 0}
        >
          確認 ({seals.length} 個印鑑)
        </Button>
      </Box>
    </Paper>
  )
}

export default MultiSealDetectionBox

