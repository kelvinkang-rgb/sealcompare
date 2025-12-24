import React, { useState, useRef, useEffect, useCallback } from 'react'
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  IconButton,
  Box,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material'
import { 
  Close as CloseIcon, 
  Image as ImageIcon, 
  CropFree as CropFreeIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  FitScreen as FitScreenIcon,
  RestartAlt as RestartAltIcon
} from '@mui/icons-material'
import { imageAPI } from '../services/api'

function ImagePreviewDialog({ open, onClose, image, sealBbox = null, seals = [], imageUrl: externalImageUrl = null }) {
  const [viewMode, setViewMode] = useState('original') // 'original' 或 'marked'
  const imgRef = useRef(null)
  const containerRef = useRef(null)
  const canvasRef = useRef(null)
  const [imageLoaded, setImageLoaded] = useState(false)
  
  // 縮放和平移狀態
  const [scale, setScale] = useState(1.0)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [isDragging, setIsDragging] = useState(false)
  const dragStartRef = useRef({ x: 0, y: 0, position: { x: 0, y: 0 } })
  const minScale = 0.5
  const maxScale = 5.0
  const zoomStep = 0.25

  // 不同印鑑使用不同顏色
  const sealColors = [
    '#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0',
    '#00BCD4', '#FFEB3B', '#795548', '#607D8B', '#E91E63'
  ]

  // 繪製標記框選到 canvas
  const drawBboxes = useCallback(() => {
    const canvas = canvasRef.current
    const img = imgRef.current
    
    if (!canvas || !img || !imageLoaded || viewMode !== 'marked') {
      return
    }

    const container = containerRef.current
    if (!container) return

    const containerRect = container.getBoundingClientRect()
    const containerWidth = containerRect.width
    const containerHeight = containerRect.height

    canvas.width = containerWidth
    canvas.height = containerHeight
    
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // 計算圖片內容的基礎顯示尺寸（考慮 objectFit: contain，不考慮用戶縮放）
    // 當使用 objectFit: contain 時，圖片會保持寬高比，實際顯示尺寸可能小於容器尺寸
    const imgAspect = img.naturalWidth / img.naturalHeight
    const containerAspect = containerWidth / containerHeight
    
    let baseDisplayWidth, baseDisplayHeight
    
    if (imgAspect > containerAspect) {
      // 圖片較寬，以容器寬度為準
      baseDisplayWidth = containerWidth
      baseDisplayHeight = containerWidth / imgAspect
    } else {
      // 圖片較高，以容器高度為準
      baseDisplayHeight = containerHeight
      baseDisplayWidth = containerHeight * imgAspect
    }
    
    // 計算基礎偏移（圖片在容器中的居中位置，不考慮用戶平移）
    const baseOffsetX = (containerWidth - baseDisplayWidth) / 2
    const baseOffsetY = (containerHeight - baseDisplayHeight) / 2

    // 計算基礎縮放比例（圖片原始顯示尺寸與原始圖片的比例）
    const baseScale = baseDisplayWidth / img.naturalWidth
    
    // 計算最終縮放比例（考慮用戶縮放）
    const currentScale = baseScale * scale

    // 獲取圖片實際顯示位置和尺寸（考慮用戶縮放和平移）
    const imgBox = img.getBoundingClientRect()
    const imgDisplayWidth = imgBox.width
    const imgDisplayHeight = imgBox.height
    const imgOffsetX = imgBox.left - containerRect.left
    const imgOffsetY = imgBox.top - containerRect.top

    // 計算用戶平移相對於基礎位置的偏移
    const panOffsetX = imgOffsetX - baseOffsetX
    const panOffsetY = imgOffsetY - baseOffsetY

    // 單印鑑模式
    if (sealBbox && seals.length === 0) {
      // 驗證 sealBbox 格式
      if (sealBbox.x === undefined || sealBbox.y === undefined || 
          sealBbox.width === undefined || sealBbox.height === undefined) {
        console.warn('ImagePreviewDialog: sealBbox 格式不正確:', sealBbox)
        return
      }

      const scaledBbox = {
        x: baseOffsetX + panOffsetX + sealBbox.x * currentScale,
        y: baseOffsetY + panOffsetY + sealBbox.y * currentScale,
        width: sealBbox.width * currentScale,
        height: sealBbox.height * currentScale
      }

      // 繪製半透明背景（整個容器）
      ctx.fillStyle = 'rgba(0, 0, 0, 0.3)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      
      // 在框選區域內繪製更淺的半透明遮罩
      ctx.fillStyle = 'rgba(0, 0, 0, 0.1)'
      ctx.fillRect(scaledBbox.x, scaledBbox.y, scaledBbox.width, scaledBbox.height)

      // 繪製邊框
      ctx.strokeStyle = '#2196F3'
      ctx.lineWidth = 3
      ctx.setLineDash([])
      ctx.strokeRect(scaledBbox.x, scaledBbox.y, scaledBbox.width, scaledBbox.height)
    }
    // 多印鑑模式
    else if (seals.length > 0) {
      // 繪製半透明背景（整個容器）
      ctx.fillStyle = 'rgba(0, 0, 0, 0.3)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      // 繪製每個印鑑框選
      seals.forEach((seal, index) => {
        // 驗證 seal 數據格式
        if (!seal || !seal.bbox) return

        const bbox = seal.bbox
        // 驗證 bbox 格式
        if (bbox.x === undefined || bbox.y === undefined || 
            bbox.width === undefined || bbox.height === undefined) {
          console.warn(`ImagePreviewDialog: 印鑑 ${index + 1} 的 bbox 格式不正確:`, bbox)
          return
        }

        const color = sealColors[index % sealColors.length]
        
        const scaledBbox = {
          x: baseOffsetX + panOffsetX + bbox.x * currentScale,
          y: baseOffsetY + panOffsetY + bbox.y * currentScale,
          width: bbox.width * currentScale,
          height: bbox.height * currentScale
        }
        
        // 在框選區域內繪製更淺的半透明遮罩
        ctx.fillStyle = 'rgba(0, 0, 0, 0.1)'
        ctx.fillRect(scaledBbox.x, scaledBbox.y, scaledBbox.width, scaledBbox.height)

        // 繪製邊框
        ctx.strokeStyle = color
        ctx.lineWidth = 3
        ctx.setLineDash([])
        ctx.strokeRect(scaledBbox.x, scaledBbox.y, scaledBbox.width, scaledBbox.height)

        // 繪製編號標籤
        ctx.fillStyle = color
        ctx.font = 'bold 18px Arial'
        ctx.fillText(
          `${index + 1}`,
          scaledBbox.x + 8,
          scaledBbox.y + 24
        )
      })
    }
  }, [imageLoaded, sealBbox, seals, sealColors, viewMode, scale, position])

  // 當圖像載入完成、視圖模式、縮放或位置變化時重新繪製
  useEffect(() => {
    if (imgRef.current?.complete && imageLoaded && viewMode === 'marked') {
      // 使用 setTimeout 確保 DOM 已更新
      setTimeout(() => {
        drawBboxes()
      }, 50)
    }
  }, [imageLoaded, viewMode, scale, position, drawBboxes])

  // 當窗口大小變化時重新繪製
  useEffect(() => {
    const handleResize = () => {
      if (imgRef.current?.complete && imageLoaded && viewMode === 'marked') {
        drawBboxes()
      }
    }
    
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [imageLoaded, viewMode, drawBboxes])

  // 縮放功能
  const handleZoom = useCallback((delta, centerX = null, centerY = null) => {
    setScale((prevScale) => {
      const newScale = Math.max(minScale, Math.min(maxScale, prevScale + delta))
      
      // 如果指定了縮放中心點，調整位置以保持該點不變
      if (centerX !== null && centerY !== null && containerRef.current && imgRef.current) {
        const container = containerRef.current
        const containerRect = container.getBoundingClientRect()
        const img = imgRef.current
        
        // 計算圖片基礎顯示位置（不考慮用戶縮放和平移）
        const imgAspect = img.naturalWidth / img.naturalHeight
        const containerAspect = containerRect.width / containerRect.height
        let baseDisplayWidth, baseDisplayHeight
        if (imgAspect > containerAspect) {
          baseDisplayWidth = containerRect.width
          baseDisplayHeight = containerRect.width / imgAspect
        } else {
          baseDisplayHeight = containerRect.height
          baseDisplayWidth = containerRect.height * imgAspect
        }
        const baseOffsetX = (containerRect.width - baseDisplayWidth) / 2
        const baseOffsetY = (containerRect.height - baseDisplayHeight) / 2
        
        // 計算鼠標位置相對於容器的座標
        const mouseX = centerX - containerRect.left
        const mouseY = centerY - containerRect.top
        
        // 計算鼠標位置相對於圖片的位置（考慮當前縮放和平移）
        // 先計算鼠標在圖片中的相對位置（相對於圖片原始尺寸）
        const imgRelativeX = (mouseX - position.x - baseOffsetX) / prevScale
        const imgRelativeY = (mouseY - position.y - baseOffsetY) / prevScale
        
        // 縮放後，調整 position 使該點保持在鼠標位置
        setPosition((prevPos) => {
          const newX = mouseX - imgRelativeX * newScale - baseOffsetX
          const newY = mouseY - imgRelativeY * newScale - baseOffsetY
          return { x: newX, y: newY }
        })
      }
      
      return newScale
    })
  }, [minScale, maxScale, position])

  // 鼠標滾輪縮放
  const handleWheel = useCallback((e) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? -zoomStep : zoomStep
    handleZoom(delta, e.clientX, e.clientY)
  }, [handleZoom, zoomStep])

  // 平移功能
  const handleMouseDown = useCallback((e) => {
    if (scale > 1.0 && e.button === 0) {
      setIsDragging(true)
      dragStartRef.current = {
        x: e.clientX,
        y: e.clientY,
        position: { ...position }
      }
    }
  }, [scale, position])

  const handleMouseMove = useCallback((e) => {
    if (isDragging && scale > 1.0 && containerRef.current && imgRef.current) {
      const deltaX = e.clientX - dragStartRef.current.x
      const deltaY = e.clientY - dragStartRef.current.y
      
      const container = containerRef.current
      const containerRect = container.getBoundingClientRect()
      const img = imgRef.current
      const imgRect = img.getBoundingClientRect()
      
      // 計算邊界限制
      const containerWidth = containerRect.width
      const containerHeight = containerRect.height
      const imgScaledWidth = imgRect.width
      const imgScaledHeight = imgRect.height
      
      // 計算允許的最大和最小位置
      const maxX = Math.max(0, (imgScaledWidth - containerWidth) / 2)
      const maxY = Math.max(0, (imgScaledHeight - containerHeight) / 2)
      const minX = -maxX
      const minY = -maxY
      
      const newX = Math.max(minX, Math.min(maxX, dragStartRef.current.position.x + deltaX))
      const newY = Math.max(minY, Math.min(maxY, dragStartRef.current.position.y + deltaY))
      
      setPosition({ x: newX, y: newY })
    }
  }, [isDragging, scale])

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  // 重置功能
  const handleReset = useCallback(() => {
    setScale(1.0)
    setPosition({ x: 0, y: 0 })
  }, [])

  // 適應窗口功能
  const handleFitToWindow = useCallback(() => {
    if (!containerRef.current || !imgRef.current || !imageLoaded) return
    
    const container = containerRef.current
    const img = imgRef.current
    const containerRect = container.getBoundingClientRect()
    const containerWidth = containerRect.width - 32 // 減去 padding
    const containerHeight = containerRect.height - 32
    
    const imgAspect = img.naturalWidth / img.naturalHeight
    const containerAspect = containerWidth / containerHeight
    
    let fitScale
    if (imgAspect > containerAspect) {
      fitScale = containerWidth / img.naturalWidth
    } else {
      fitScale = containerHeight / img.naturalHeight
    }
    
    // 限制在最小和最大縮放比例之間
    fitScale = Math.max(minScale, Math.min(maxScale, fitScale * 0.9)) // 0.9 是邊距
    
    setScale(fitScale)
    setPosition({ x: 0, y: 0 })
  }, [imageLoaded, minScale, maxScale])

  // 當對話框打開時重置狀態
  useEffect(() => {
    if (open) {
      setImageLoaded(false)
      setViewMode('original')
      setScale(1.0)
      setPosition({ x: 0, y: 0 })
      setIsDragging(false)
    }
  }, [open])

  // 如果提供了外部 imageUrl，優先使用它；否則從 image 對象獲取
  const imageUrl = externalImageUrl || (image ? imageAPI.getFile(image.id) : null)
  
  // 如果既沒有 image 也沒有 imageUrl，不顯示對話框
  if (!image && !imageUrl) {
    return null
  }

  const hasSeal = (sealBbox !== null && sealBbox !== undefined) || (seals && seals.length > 0)

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: {
          maxWidth: '95vw',
          maxHeight: '95vh',
        },
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 1 }}>
          <Typography variant="h6">
            {image?.filename || '圖片預覽'}
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
            {/* 縮放工具欄 */}
            <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center', border: '1px solid', borderColor: 'divider', borderRadius: 1, p: 0.5 }}>
              <IconButton
                size="small"
                onClick={() => handleZoom(-zoomStep)}
                disabled={scale <= minScale}
                title="縮小"
              >
                <ZoomOutIcon fontSize="small" />
              </IconButton>
              <Typography variant="body2" sx={{ minWidth: '50px', textAlign: 'center', px: 1 }}>
                {Math.round(scale * 100)}%
              </Typography>
              <IconButton
                size="small"
                onClick={() => handleZoom(zoomStep)}
                disabled={scale >= maxScale}
                title="放大"
              >
                <ZoomInIcon fontSize="small" />
              </IconButton>
              <IconButton
                size="small"
                onClick={handleFitToWindow}
                title="適應窗口"
              >
                <FitScreenIcon fontSize="small" />
              </IconButton>
              <IconButton
                size="small"
                onClick={handleReset}
                disabled={scale === 1.0 && position.x === 0 && position.y === 0}
                title="重置"
              >
                <RestartAltIcon fontSize="small" />
              </IconButton>
            </Box>
            
            {hasSeal && (
              <ToggleButtonGroup
                value={viewMode}
                exclusive
                onChange={(e, newMode) => {
                  if (newMode !== null) {
                    setViewMode(newMode)
                  }
                }}
                size="small"
              >
                <ToggleButton value="original">
                  <ImageIcon sx={{ mr: 0.5 }} fontSize="small" />
                  原始圖片
                </ToggleButton>
                <ToggleButton value="marked">
                  <CropFreeIcon sx={{ mr: 0.5 }} fontSize="small" />
                  標記框選
                </ToggleButton>
              </ToggleButtonGroup>
            )}
            <IconButton
              aria-label="close"
              onClick={onClose}
              sx={{
                color: (theme) => theme.palette.grey[500],
              }}
            >
              <CloseIcon />
            </IconButton>
          </Box>
        </Box>
      </DialogTitle>
      <DialogContent>
        <Box
          ref={containerRef}
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          sx={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: '60vh',
            maxHeight: '80vh',
            backgroundColor: '#f5f5f5',
            p: 2,
            position: 'relative',
            overflow: 'hidden',
            cursor: scale > 1.0 ? (isDragging ? 'grabbing' : 'grab') : 'default',
            userSelect: 'none',
          }}
        >
          <Box
            sx={{
              position: 'relative',
              display: 'inline-block',
              maxWidth: '100%',
              maxHeight: '80vh',
              transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
              transformOrigin: 'center center',
              transition: isDragging ? 'none' : 'transform 0.2s ease-out',
            }}
          >
            <img
              ref={imgRef}
              src={imageUrl}
              alt={image?.filename || '圖片'}
              style={{
                maxWidth: '100%',
                maxHeight: '80vh',
                objectFit: 'contain',
                borderRadius: '4px',
                display: 'block',
                pointerEvents: 'none',
              }}
              onLoad={() => {
                setImageLoaded(true)
              }}
              onError={(e) => {
                e.target.style.display = 'none'
              }}
              draggable={false}
            />
            {viewMode === 'marked' && hasSeal && imageLoaded && (
              <canvas
                ref={canvasRef}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  pointerEvents: 'none',
                }}
              />
            )}
          </Box>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>關閉</Button>
      </DialogActions>
    </Dialog>
  )
}

export default ImagePreviewDialog

