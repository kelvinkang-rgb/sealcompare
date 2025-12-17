import React from 'react'
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  Divider,
  Paper,
} from '@mui/material'
import { Info as InfoIcon } from '@mui/icons-material'

function MetricsExplanationDialog({ open, onClose }) {
  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <InfoIcon color="primary" />
          比對指標說明
        </Box>
      </DialogTitle>
      <DialogContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, mt: 1 }}>
          {/* SSIM 說明 */}
          <Paper sx={{ p: 2, backgroundColor: '#f9f9f9' }}>
            <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
              SSIM (結構相似性指數)
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              <strong>全名：</strong>Structural Similarity Index
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              <strong>範圍：</strong>0-1（1 表示完全相同）
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>計算方式：</strong>
              <br />
              比較兩個圖像的亮度、對比度、結構三個方面。使用高斯模糊計算局部統計量（均值、方差、協方差），綜合評估結構相似性。
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>特點：</strong>
              <br />
              • 符合人眼感知，能捕捉結構相似性
              <br />
              • 對亮度變化較不敏感
              <br />
              • 適合評估整體相似度
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>在印鑑比對中的作用：</strong>
              <br />
              評估整體結構是否一致。即使顏色略有差異，只要結構相似，仍能獲得高分。
            </Typography>
          </Paper>

          {/* 模板匹配說明 */}
          <Paper sx={{ p: 2, backgroundColor: '#f9f9f9' }}>
            <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
              模板匹配 (Template Matching)
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              <strong>範圍：</strong>0-1（1 表示完全匹配）
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>計算方式：</strong>
              <br />
              使用 OpenCV 的模板匹配方法，將圖像2作為模板，在圖像1中搜索最佳匹配位置。使用歸一化相關系數（TM_CCOEFF_NORMED）計算匹配度。
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>特點：</strong>
              <br />
              • 對位置和形狀敏感
              <br />
              • 能找出最佳對齊位置
              <br />
              • 計算速度較快
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>在印鑑比對中的作用：</strong>
              <br />
              評估圖案是否能在對齊後匹配。如果兩個印章的圖案相同，即使初始位置不同，模板匹配也能找到最佳對齊位置並給出高分。
            </Typography>
          </Paper>

          {/* 像素差異說明 */}
          <Paper sx={{ p: 2, backgroundColor: '#f9f9f9' }}>
            <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
              像素差異 (Pixel Difference)
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              <strong>範圍：</strong>0-1（0 表示完全相同，1 表示完全不同）
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>計算方式：</strong>
              <br />
              逐像素比較兩個圖像的絕對差值，統計有差異的像素數量。差異率 = 有差異的像素數 / 總像素數。
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>特點：</strong>
              <br />
              • 最直觀的差異衡量方式
              <br />
              • 對每個像素都敏感
              <br />
              • 能反映細微差異
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>在印鑑比對中的作用：</strong>
              <br />
              識別具體差異位置，量化差異程度。數值越小表示差異越小，兩個印章越相似。
            </Typography>
            <Box
              sx={{
                mt: 1.5,
                p: 1.5,
                backgroundColor: '#fff3cd',
                borderRadius: 1,
                border: '1px solid #ffc107',
              }}
            >
              <Typography variant="body2" sx={{ fontWeight: 'bold', color: '#856404' }}>
                ⚠️ 注意：此指標顯示的是差異率，數值越小越好！
              </Typography>
            </Box>
          </Paper>

          <Divider />

          {/* 綜合評分說明 */}
          <Paper sx={{ p: 2, backgroundColor: '#e3f2fd' }}>
            <Typography variant="h6" gutterBottom sx={{ color: 'primary.main' }}>
              綜合評分方式
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              系統使用加權平均計算最終相似度：
            </Typography>
            <Box
              sx={{
                mt: 1.5,
                p: 2,
                backgroundColor: 'white',
                borderRadius: 1,
                border: '1px solid #e0e0e0',
              }}
            >
              <Typography
                variant="body1"
                sx={{ fontFamily: 'monospace', textAlign: 'center', fontWeight: 'bold' }}
              >
                最終相似度 = SSIM × 50% + 模板匹配 × 30% + (1 - 像素差異) × 20%
              </Typography>
            </Box>
            <Typography variant="body2" sx={{ mt: 1.5 }}>
              <strong>權重說明：</strong>
              <br />
              • <strong>SSIM 50%</strong>：重視整體結構相似性
              <br />
              • <strong>模板匹配 30%</strong>：重視對齊後的匹配度
              <br />
              • <strong>像素差異 20%</strong>：考慮細微差異
            </Typography>
          </Paper>

          {/* 實際應用建議 */}
          <Paper sx={{ p: 2, backgroundColor: '#f1f8e9' }}>
            <Typography variant="h6" gutterBottom sx={{ color: 'success.main' }}>
              如何解讀結果
            </Typography>
            <Typography variant="body2" sx={{ mt: 1 }}>
              <strong>三個指標都高（SSIM 高、模板匹配高、像素差異低）：</strong>
              <br />
              表示兩個印章非常相似，很可能是同一枚印章。
            </Typography>
            <Typography variant="body2" sx={{ mt: 1.5 }}>
              <strong>SSIM 高但模板匹配低：</strong>
              <br />
              結構相似但對齊不佳，可能需要調整旋轉角度。
            </Typography>
            <Typography variant="body2" sx={{ mt: 1.5 }}>
              <strong>像素差異高：</strong>
              <br />
              存在明顯差異，可能是不同的印章或印章有損壞。
            </Typography>
          </Paper>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} variant="contained">
          關閉
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default MetricsExplanationDialog

