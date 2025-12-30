import React, { useEffect, useMemo, useState } from 'react'
import { Box, CircularProgress, TextField, MenuItem, Typography } from '@mui/material'
import { useQuery } from '@tanstack/react-query'
import { imageAPI } from '../services/api'

/**
 * PdfPagePicker
 * - 接收後端 upload/get 回傳的 PDF root image（含 pages[]）
 * - 讓使用者選頁，並自動拉回該頁 page image 的完整資料（含 seal_bbox / multiple_seals）
 */
function PdfPagePicker({
  pdfImage,
  preferredPageImageId = null,
  label = 'PDF 頁面',
  disabled = false,
  onPageImageLoaded,
}) {
  const pages = Array.isArray(pdfImage?.pages) ? pdfImage.pages : []

  const defaultPageId = useMemo(() => {
    if (preferredPageImageId) return preferredPageImageId
    return pages[0]?.id || null
  }, [pages, preferredPageImageId])

  const [selectedPageId, setSelectedPageId] = useState(defaultPageId)

  useEffect(() => {
    setSelectedPageId(defaultPageId)
  }, [defaultPageId])

  const pageQuery = useQuery({
    queryKey: ['pdf-page-image', selectedPageId],
    queryFn: () => imageAPI.get(selectedPageId),
    enabled: !!selectedPageId,
    refetchOnWindowFocus: false,
  })

  useEffect(() => {
    if (pageQuery.data && onPageImageLoaded) {
      onPageImageLoaded(pageQuery.data)
    }
  }, [pageQuery.data, onPageImageLoaded])

  if (!pdfImage?.is_pdf) return null
  if (!pages.length) {
    return (
      <Box sx={{ mt: 1 }}>
        <Typography variant="caption" color="text.secondary">
          未取得 PDF 分頁資訊
        </Typography>
      </Box>
    )
  }

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
      <TextField
        select
        size="small"
        label={label}
        value={selectedPageId || ''}
        onChange={(e) => setSelectedPageId(e.target.value)}
        disabled={disabled}
        sx={{ minWidth: 220 }}
      >
        {pages.map((p) => (
          <MenuItem key={p.id} value={p.id}>
            第 {p.page_index} 頁
          </MenuItem>
        ))}
      </TextField>
      {pageQuery.isFetching && <CircularProgress size={18} />}
    </Box>
  )
}

export default PdfPagePicker


