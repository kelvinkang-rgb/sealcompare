import React from 'react'
import { useNavigate } from 'react-router-dom'
import { Container, Typography, Box, Card, CardContent, Grid } from '@mui/material'
import { BarChart, Science } from '@mui/icons-material'

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

        <Grid item xs={12} md={4}>
          <Card sx={{ height: '100%', cursor: 'pointer' }} onClick={() => navigate('/multi-seal-test')}>
            <CardContent sx={{ textAlign: 'center', py: 4 }}>
              <Science sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
              <Typography variant="h5" gutterBottom>
                多印鑑測試
              </Typography>
              <Typography variant="body2" color="text.secondary">
                測試多印鑑檢測與比對功能
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  )
}

export default Home

