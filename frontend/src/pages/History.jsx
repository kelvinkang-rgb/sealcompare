import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Container,
  Typography,
  Button,
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Snackbar,
  Alert,
} from '@mui/material'
import { Edit as EditIcon, Delete as DeleteIcon } from '@mui/icons-material'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { comparisonAPI } from '../services/api'
import ComparisonEditDialog from '../components/ComparisonEditDialog'
import DeleteConfirmDialog from '../components/DeleteConfirmDialog'
import { useFeatureFlag, FEATURE_FLAGS } from '../config/featureFlags'

function History() {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [editDialogOpen, setEditDialogOpen] = useState(false)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  const [selectedComparison, setSelectedComparison] = useState(null)
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' })
  
  // 功能開關
  const showComparisonEditDialog = useFeatureFlag(FEATURE_FLAGS.COMPARISON_EDIT_DIALOG)
  const showDeleteConfirmDialog = useFeatureFlag(FEATURE_FLAGS.DELETE_CONFIRM_DIALOG)

  const { data: comparisons, isLoading } = useQuery({
    queryKey: ['comparisons'],
    queryFn: () => comparisonAPI.list(),
  })

  const updateMutation = useMutation({
    mutationFn: ({ comparisonId, data }) => comparisonAPI.update(comparisonId, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['comparisons'])
      setEditDialogOpen(false)
      setSelectedComparison(null)
      setSnackbar({ open: true, message: '比對記錄已更新', severity: 'success' })
    },
    onError: (error) => {
      setSnackbar({
        open: true,
        message: error?.response?.data?.detail || '更新失敗',
        severity: 'error',
      })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (comparisonId) => comparisonAPI.delete(comparisonId),
    onSuccess: () => {
      queryClient.invalidateQueries(['comparisons'])
      setDeleteDialogOpen(false)
      setSelectedComparison(null)
      setSnackbar({ open: true, message: '比對記錄已刪除', severity: 'success' })
    },
    onError: (error) => {
      setSnackbar({
        open: true,
        message: error?.response?.data?.detail || '刪除失敗',
        severity: 'error',
      })
    },
  })

  const handleEdit = (comparison) => {
    setSelectedComparison(comparison)
    setEditDialogOpen(true)
  }

  const handleDelete = (comparison) => {
    setSelectedComparison(comparison)
    setDeleteDialogOpen(true)
  }

  const handleSaveEdit = (updateData) => {
    if (selectedComparison) {
      updateMutation.mutate({
        comparisonId: selectedComparison.id,
        data: updateData,
      })
    }
  }

  const handleConfirmDelete = () => {
    if (selectedComparison) {
      deleteMutation.mutate(selectedComparison.id)
    }
  }

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false })
  }

  if (isLoading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Typography>載入中...</Typography>
      </Container>
    )
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Button onClick={() => navigate('/')} sx={{ mb: 2 }}>
          返回首頁
        </Button>
        <Typography variant="h4" component="h1" gutterBottom>
          比對記錄
        </Typography>
      </Box>

      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>狀態</TableCell>
              <TableCell>創建時間</TableCell>
              <TableCell>操作</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {comparisons?.map((comparison) => (
              <TableRow key={comparison.id}>
                <TableCell>{comparison.id.slice(0, 8)}...</TableCell>
                <TableCell>
                  <Chip
                    label={comparison.status}
                    color={
                      comparison.status === 'completed'
                        ? 'success'
                        : comparison.status === 'failed'
                        ? 'error'
                        : 'default'
                    }
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  {new Date(comparison.created_at).toLocaleString('zh-TW')}
                </TableCell>
                <TableCell>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={() => navigate(`/compare?comparisonId=${comparison.id}`)}
                    >
                      查看
                    </Button>
                    {showComparisonEditDialog && (
                      <IconButton
                        size="small"
                        color="primary"
                        onClick={() => handleEdit(comparison)}
                        title="編輯"
                      >
                        <EditIcon fontSize="small" />
                      </IconButton>
                    )}
                    {showDeleteConfirmDialog && (
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => handleDelete(comparison)}
                        title="刪除"
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    )}
                  </Box>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* 編輯對話框 */}
      {showComparisonEditDialog && (
        <ComparisonEditDialog
          open={editDialogOpen}
          onClose={() => {
            setEditDialogOpen(false)
            setSelectedComparison(null)
          }}
          comparison={selectedComparison}
          onSave={handleSaveEdit}
        />
      )}

      {/* 刪除確認對話框 */}
      {showDeleteConfirmDialog && (
        <DeleteConfirmDialog
          open={deleteDialogOpen}
          onClose={() => {
            setDeleteDialogOpen(false)
            setSelectedComparison(null)
          }}
          onConfirm={handleConfirmDelete}
          comparison={selectedComparison}
        />
      )}

      {/* 操作反饋提示 */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={3000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert
          onClose={handleCloseSnackbar}
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Container>
  )
}

export default History

