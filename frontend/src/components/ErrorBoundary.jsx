import React from 'react'
import { Box, Button, Paper, Typography } from '@mui/material'

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null, errorInfo: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, errorInfo) {
    // 確保就算 UI 白屏也能在 console 看到錯誤線索
    // eslint-disable-next-line no-console
    console.error('ErrorBoundary caught error:', error, errorInfo)
    this.setState({ error, errorInfo })
  }

  render() {
    if (!this.state.hasError) return this.props.children

    const message = this.state.error?.message || '發生未知錯誤'

    return (
      <Box sx={{ p: 2 }}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            畫面發生錯誤，已停止渲染
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            {message}
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Button variant="contained" onClick={() => window.location.reload()}>
              重新整理
            </Button>
            <Button variant="outlined" onClick={() => (window.location.href = '/')}>
              回首頁
            </Button>
          </Box>
          {this.props.showDetails && this.state.errorInfo?.componentStack && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="caption" color="text.secondary" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                {this.state.errorInfo.componentStack}
              </Typography>
            </Box>
          )}
        </Paper>
      </Box>
    )
  }
}


