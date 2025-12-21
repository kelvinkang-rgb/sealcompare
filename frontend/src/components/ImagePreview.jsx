import React, { useState, useRef, useEffect } from 'react'
import { Box, Paper, Typography, IconButton, Chip } from '@mui/material'
import { Edit as EditIcon, CheckCircle as CheckCircleIcon } from '@mui/icons-material'
import { imageAPI } from '../services/api'

function ImagePreview({ image, label, onEdit, showSealIndicator = true }) {
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 })
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 })
  const [croppedImageUrl, setCroppedImageUrl] = useState(null)
  const imgRef = useRef(null)
  const containerRef = useRef(null)

  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setContainerSize({
          width: containerRef.current.offsetWidth,
          height: containerRef.current.offsetHeight,
        })
      }
    }
    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [])

  useEffect(() => {
    if (imgRef.current && imgRef.current.complete) {
      setImageSize({
        width: imgRef.current.naturalWidth,
        height: imgRef.current.naturalHeight,
      })
    }
  }, [image])

  // 裁切圖像
  useEffect(() => {
    const cropImage = async () => {
      if (!image || !image.seal_bbox) {
        setCroppedImageUrl(null)
        return
      }

      const img = new Image()
      img.crossOrigin = 'anonymous'
      img.src = imageAPI.getFile(image.id)

      img.onload = () => {
        const canvas = document.createElement('canvas')
        const ctx = canvas.getContext('2d')
        
        const bbox = image.seal_bbox
        // 確保裁切區域在圖像範圍內
        const x = Math.max(0, Math.min(bbox.x, img.width - 1))
        const y = Math.max(0, Math.min(bbox.y, img.height - 1))
        const width = Math.min(bbox.width, img.width - x)
        const height = Math.min(bbox.height, img.height - y)

        if (width > 0 && height > 0) {
          canvas.width = width
          canvas.height = height
          
          // 裁切圖像
          ctx.drawImage(
            img,
            x, y, width, height,
            0, 0, width, height
          )
          
          // 轉換為 URL
          const croppedUrl = canvas.toDataURL('image/jpeg', 0.95)
          setCroppedImageUrl(croppedUrl)
        } else {
          setCroppedImageUrl(null)
        }
      }

      img.onerror = () => {
        setCroppedImageUrl(null)
      }
    }

    cropImage()
  }, [image])

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
  
  // 使用裁切後的圖像 URL（如果有）
  const displayImageUrl = croppedImageUrl || imageUrl
  
  // 計算顯示圖像的尺寸（如果是裁切後的圖像，使用裁切區域的尺寸）
  const getDisplayImageSize = () => {
    if (croppedImageUrl && image.seal_bbox) {
      return {
        width: image.seal_bbox.width,
        height: image.seal_bbox.height,
      }
    }
    return imageSize
  }

  const displaySize = getDisplayImageSize()

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
        sx={{
          position: 'relative',
          width: '100%',
          paddingTop: '75%', // 4:3 比例
          backgroundColor: '#f5f5f5',
          borderRadius: 1,
          overflow: 'hidden',
          border: hasSeal ? '2px solid #4caf50' : '1px solid #ddd',
        }}
      >
        <img
          ref={imgRef}
          src={displayImageUrl}
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
            if (!croppedImageUrl) {
              // 只有當不是裁切圖像時才更新原始圖像尺寸
              setImageSize({
                width: e.target.naturalWidth,
                height: e.target.naturalHeight,
              })
            }
          }}
        />
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

