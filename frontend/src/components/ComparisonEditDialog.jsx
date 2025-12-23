import React, { useState, useEffect } from 'react'
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Box,
  Alert,
} from '@mui/material'

function ComparisonEditDialog({ open, onClose, comparison, onSave }) {
  const [notes, setNotes] = useState('')
  const [error, setError] = useState(null)

  useEffect(() => {
    if (comparison) {
      setNotes(comparison.notes || '')
      setError(null)
    }
  }, [comparison, open])

  const handleSave = () => {
    const updateData = {
      notes: notes.trim() || null,
    }

    onSave(updateData)
  }

  const handleClose = () => {
    setError(null)
    onClose()
  }

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>編輯比對記錄</DialogTitle>
      <DialogContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
          {error && (
            <Alert severity="error" onClose={() => setError(null)}>
              {error}
            </Alert>
          )}

          <TextField
            label="備註/標籤"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            multiline
            rows={3}
            fullWidth
            placeholder="輸入備註或標籤..."
            helperText="可選：添加備註或標籤以便管理"
          />
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose}>取消</Button>
        <Button onClick={handleSave} variant="contained">
          保存
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default ComparisonEditDialog

