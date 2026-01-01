import React from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import MultiSealTest from './pages/MultiSealTest'
import ErrorBoundary from './components/ErrorBoundary'

// 確保 React Router 的 useSearchParams 可以正常工作

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
})

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Routes>
          <Route path="/" element={<Navigate to="/multi-seal-test" replace />} />
          <Route
            path="/multi-seal-test"
            element={
              <ErrorBoundary>
                <MultiSealTest />
              </ErrorBoundary>
            }
          />
        </Routes>
      </Router>
    </ThemeProvider>
  )
}

export default App

