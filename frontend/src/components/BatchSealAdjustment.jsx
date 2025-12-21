import React, { useState } from 'react'
import { Box, Paper, Typography, Button, Grid, Dialog, DialogTitle, DialogContent, DialogActions } from '@mui/material'
import SealDetectionBox from './SealDetectionBox'

function BatchSealAdjustment({ 
  image1Id, 
  image2Id, 
  image1InitialBbox, 
  image1InitialCenter,
  image2InitialBbox, 
  image2InitialCenter,
  onConfirm,
  onCancel 
}) {
  const [image1Bbox, setImage1Bbox] = useState(image1InitialBbox)
  const [image1Center, setImage1Center] = useState(image1InitialCenter)
  const [image2Bbox, setImage2Bbox] = useState(image2InitialBbox)
  const [image2Center, setImage2Center] = useState(image2InitialCenter)
  const [image1Confirmed, setImage1Confirmed] = useState(false)
  const [image2Confirmed, setImage2Confirmed] = useState(false)

  const handleImage1Confirm = (locationData) => {
    setImage1Bbox(locationData.bbox)
    setImage1Center(locationData.center)
    setImage1Confirmed(true)
  }

  const handleImage2Confirm = (locationData) => {
    setImage2Bbox(locationData.bbox)
    setImage2Center(locationData.center)
    setImage2Confirmed(true)
  }

  const handleBatchConfirm = () => {
    if (onConfirm) {
      onConfirm({
        image1: {
          bbox: image1Bbox,
          center: image1Center
        },
        image2: {
          bbox: image2Bbox,
          center: image2Center
        }
      })
    }
  }

  const canConfirm = image1Bbox && image2Bbox

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        批量調整印鑑位置
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        在一個視圖中同時調整兩個圖像的印鑑位置，方便對比和調整
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, border: image1Confirmed ? '2px solid #4caf50' : '1px solid #ddd' }}>
            <Typography variant="subtitle1" gutterBottom fontWeight="bold">
              圖像1
            </Typography>
            {image1Id && (
              <SealDetectionBox
                imageId={image1Id}
                initialBbox={image1InitialBbox}
                initialCenter={image1InitialCenter}
                onConfirm={handleImage1Confirm}
                onCancel={() => {}}
                showCancelButton={false}
              />
            )}
            {image1Confirmed && (
              <Box sx={{ mt: 1, p: 1, bgcolor: '#e8f5e9', borderRadius: 1 }}>
                <Typography variant="caption" color="success.main">
                  ✓ 圖像1已確認
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2, border: image2Confirmed ? '2px solid #4caf50' : '1px solid #ddd' }}>
            <Typography variant="subtitle1" gutterBottom fontWeight="bold">
              圖像2
            </Typography>
            {image2Id && (
              <SealDetectionBox
                imageId={image2Id}
                initialBbox={image2InitialBbox}
                initialCenter={image2InitialCenter}
                onConfirm={handleImage2Confirm}
                onCancel={() => {}}
                showCancelButton={false}
              />
            )}
            {image2Confirmed && (
              <Box sx={{ mt: 1, p: 1, bgcolor: '#e8f5e9', borderRadius: 1 }}>
                <Typography variant="caption" color="success.main">
                  ✓ 圖像2已確認
                </Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>

      <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
        {onCancel && (
          <Button variant="outlined" onClick={onCancel}>
            取消
          </Button>
        )}
        <Button 
          variant="contained" 
          onClick={handleBatchConfirm}
          disabled={!canConfirm}
        >
          確認全部
        </Button>
      </Box>
    </Box>
  )
}

export default BatchSealAdjustment

