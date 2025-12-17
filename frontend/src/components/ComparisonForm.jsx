import React, { useState, forwardRef, useImperativeHandle } from 'react'
import { Button, Box, Alert } from '@mui/material'
import { useMutation } from '@tanstack/react-query'
import { imageAPI } from '../services/api'

const ComparisonForm = forwardRef(({ onSubmit }, ref) => {
  const [image1, setImage1] = useState(null)
  const [image2, setImage2] = useState(null)
  const [enableRotation, setEnableRotation] = useState(true)

  const uploadImage1Mutation = useMutation({
    mutationFn: imageAPI.upload,
  })

  const uploadImage2Mutation = useMutation({
    mutationFn: imageAPI.upload,
  })

  // 暴露 reset 方法給父組件
  useImperativeHandle(ref, () => ({
    reset: () => {
      setImage1(null)
      setImage2(null)
      setEnableRotation(true)
      uploadImage1Mutation.reset()
      uploadImage2Mutation.reset()
      // 重置文件輸入元素
      const image1Input = document.getElementById('image1-upload')
      const image2Input = document.getElementById('image2-upload')
      if (image1Input) image1Input.value = ''
      if (image2Input) image2Input.value = ''
    },
  }))

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
      uploadImage1Mutation.data?.id &&
      uploadImage2Mutation.data?.id
    ) {
      onSubmit(
        uploadImage1Mutation.data.id,
        uploadImage2Mutation.data.id,
        enableRotation
      )
    }
  }

  return (
    <Box component="form" onSubmit={handleSubmit}>
      <Box sx={{ mb: 2 }}>
        <input
          accept="image/*"
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
            disabled={uploadImage1Mutation.isPending}
          >
            {uploadImage1Mutation.isPending
              ? '上傳中...'
              : uploadImage1Mutation.data
              ? `圖像1: ${uploadImage1Mutation.data.filename}`
              : '選擇圖像1'}
          </Button>
        </label>
        {uploadImage1Mutation.isError && (
          <Alert severity="error" sx={{ mt: 1 }}>
            上傳失敗
          </Alert>
        )}
      </Box>

      <Box sx={{ mb: 2 }}>
        <input
          accept="image/*"
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
            disabled={uploadImage2Mutation.isPending}
          >
            {uploadImage2Mutation.isPending
              ? '上傳中...'
              : uploadImage2Mutation.data
              ? `圖像2: ${uploadImage2Mutation.data.filename}`
              : '選擇圖像2'}
          </Button>
        </label>
        {uploadImage2Mutation.isError && (
          <Alert severity="error" sx={{ mt: 1 }}>
            上傳失敗
          </Alert>
        )}
      </Box>

      <Button
        type="submit"
        variant="contained"
        fullWidth
        disabled={
          !uploadImage1Mutation.data ||
          !uploadImage2Mutation.data ||
          uploadImage1Mutation.isPending ||
          uploadImage2Mutation.isPending
        }
      >
        開始比對
      </Button>
    </Box>
  )
})

ComparisonForm.displayName = 'ComparisonForm'

export default ComparisonForm

