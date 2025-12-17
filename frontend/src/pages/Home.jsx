import React from 'react'
import { useNavigate } from 'react-router-dom'
import { Container, Typography, Button, Box, Card, CardContent, Grid } from '@mui/material'
import { CompareArrows, History, BarChart } from '@mui/icons-material'

function Home() {
  const navigate = useNavigate()

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <Typography variant="h3" component="h1" gutterBottom>
          印鑑比對系統
        </Typography>
        <Typography variant="h6" color="text.secondary">
          快速、準確的印章圖像比對工具
        </Typography>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', cursor: 'pointer' }} onClick={() => navigate('/compare')}>
            <CardContent sx={{ textAlign: 'center', py: 4 }}>
              <CompareArrows sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
              <Typography variant="h5" gutterBottom>
                開始比對
              </Typography>
              <Typography variant="body2" color="text.secondary">
                上傳兩個印章圖像進行比對
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', cursor: 'pointer' }} onClick={() => navigate('/history')}>
            <CardContent sx={{ textAlign: 'center', py: 4 }}>
              <History sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
              <Typography variant="h5" gutterBottom>
                比對記錄
              </Typography>
              <Typography variant="body2" color="text.secondary">
                查看歷史比對結果
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent sx={{ textAlign: 'center', py: 4 }}>
              <BarChart sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
              <Typography variant="h5" gutterBottom>
                統計資訊
              </Typography>
              <Typography variant="body2" color="text.secondary">
                查看系統統計數據
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Box sx={{ textAlign: 'center', mt: 4 }}>
        <Button
          variant="contained"
          size="large"
          onClick={() => navigate('/compare')}
          sx={{ px: 4 }}
        >
          開始使用
        </Button>
      </Box>
    </Container>
  )
}

export default Home

