import React from 'react'
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
  Box,
  Typography,
} from '@mui/material'
import { Warning as WarningIcon } from '@mui/icons-material'

function DeleteConfirmDialog({ open, onClose, onConfirm, comparison }) {
  return (
    <Dialog open={open} onClose={onClose}>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <WarningIcon color="warning" />
          確認刪除
        </Box>
      </DialogTitle>
      <DialogContent>
        <DialogContentText>
          您確定要刪除此比對記錄嗎？
        </DialogContentText>
        {comparison && (
          <Box sx={{ mt: 2, p: 2, backgroundColor: '#f5f5f5', borderRadius: 1 }}>
            <Typography variant="body2" color="text.secondary">
              <strong>記錄 ID:</strong> {comparison.id.slice(0, 8)}...
            </Typography>
            {comparison.similarity && (
              <Typography variant="body2" color="text.secondary">
                <strong>相似度:</strong> {(comparison.similarity * 100).toFixed(2)}%
              </Typography>
            )}
            <Typography variant="body2" color="text.secondary">
              <strong>創建時間:</strong>{' '}
              {new Date(comparison.created_at).toLocaleString('zh-TW')}
            </Typography>
          </Box>
        )}
        <DialogContentText sx={{ mt: 2, color: 'text.secondary', fontSize: '0.875rem' }}>
          注意：此操作將標記記錄為已刪除，記錄不會從系統中完全移除，但將不再顯示在列表中。
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>取消</Button>
        <Button onClick={onConfirm} color="error" variant="contained">
          確認刪除
        </Button>
      </DialogActions>
    </Dialog>
  )
}

export default DeleteConfirmDialog

