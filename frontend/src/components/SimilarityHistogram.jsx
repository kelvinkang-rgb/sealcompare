import React, { useMemo, useState } from 'react'
import { Paper, Typography, Box, TextField, Slider } from '@mui/material'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

function SimilarityHistogram({ results }) {
  // 間隔設定（百分比），默認5%
  const [binSize, setBinSize] = useState(5)

  // 計算 histogram 數據
  const histogramData = useMemo(() => {
    // 根據間隔動態計算區間數量
    const numBins = Math.ceil(100 / binSize)
    const bins = Array.from({ length: numBins }, (_, i) => {
      const minPercent = i * binSize
      const maxPercent = Math.min((i + 1) * binSize, 100)
      return {
        range: `${minPercent}-${maxPercent}%`,
        min: minPercent / 100,
        max: maxPercent / 100,
        count: 0,
        index: i
      }
    })

    // 統計每個區間的數量
    if (results && results.length > 0) {
      results.forEach(result => {
        if (result.mask_based_similarity !== null && result.mask_based_similarity !== undefined) {
          const similarity = result.mask_based_similarity
          const similarityPercent = similarity * 100
          // 找到對應的區間
          const binIndex = Math.min(
            Math.floor(similarityPercent / binSize),
            numBins - 1 // 確保不超過最後一個區間
          )
          if (binIndex >= 0 && binIndex < bins.length) {
            bins[binIndex].count++
          }
        }
      })
    }

    return bins
  }, [results, binSize])

  // 計算總數
  const totalCount = histogramData.reduce((sum, bin) => sum + bin.count, 0)

  if (totalCount === 0) {
    return (
      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          Mask相似度分布統計
        </Typography>
        <Typography variant="body2" color="text.secondary">
          暫無數據
        </Typography>
      </Paper>
    )
  }

  return (
    <Paper sx={{ p: 3, mt: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">
          Mask相似度分布統計
        </Typography>
      </Box>

      {/* 間隔設定 */}
      <Box sx={{ mb: 2, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
        <Typography variant="body2" gutterBottom fontWeight="bold">
          區間間隔設定
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
          設定每個區間的寬度（當前: {binSize}%）
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Slider
            value={binSize}
            onChange={(e, value) => setBinSize(value)}
            min={1}
            max={20}
            step={1}
            marks={[
              { value: 1, label: '1%' },
              { value: 5, label: '5%' },
              { value: 10, label: '10%' },
              { value: 20, label: '20%' }
            ]}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => `${value}%`}
            sx={{ flex: 1 }}
          />
          <TextField
            type="number"
            value={binSize}
            onChange={(e) => {
              const val = parseInt(e.target.value)
              if (!isNaN(val) && val >= 1 && val <= 20) {
                setBinSize(val)
              }
            }}
            inputProps={{ 
              min: 1, 
              max: 20, 
              step: 1 
            }}
            size="small"
            sx={{ width: '100px' }}
            label="間隔%"
          />
        </Box>
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
          共 {histogramData.length} 個區間
        </Typography>
      </Box>

      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={histogramData}
          margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="range" 
            angle={-45}
            textAnchor="end"
            height={80}
            interval={histogramData.length > 20 ? 'preserveStartEnd' : 0}
          />
          <YAxis 
            label={{ value: '印鑑數量', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip 
            formatter={(value) => [`${value} 個印鑑`, '數量']}
            labelFormatter={(label) => `相似度: ${label}`}
          />
          <Bar dataKey="count" fill="#1976d2" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>

      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
        總計: {totalCount} 個印鑑
      </Typography>
    </Paper>
  )
}

export default SimilarityHistogram

