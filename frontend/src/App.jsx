import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import Home from './pages/Home'
import Comparison from './pages/Comparison'
import History from './pages/History'
import MultiSealTest from './pages/MultiSealTest'

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
          <Route path="/" element={<Home />} />
          <Route path="/compare" element={<Comparison />} />
          <Route path="/history" element={<History />} />
          <Route path="/multi-seal-test" element={<MultiSealTest />} />
        </Routes>
      </Router>
    </ThemeProvider>
  )
}

export default App

