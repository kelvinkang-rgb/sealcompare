import React from 'react'
import {
  Box,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Typography,
  LinearProgress,
  Chip,
  Paper,
} from '@mui/material'
import {
  CheckCircle as CheckCircleIcon,
  RadioButtonUnchecked as RadioButtonUncheckedIcon,
  HourglassEmpty as HourglassEmptyIcon,
  Error as ErrorIcon,
} from '@mui/icons-material'

function ProcessingStages({ processingStages }) {
  if (!processingStages || !processingStages.stages) {
    return null
  }
  
  // 調試：檢查 processing_stages 結構
  console.log('DEBUG ProcessingStages 完整數據:', JSON.stringify(processingStages, null, 2))

  // 計算總時間（用於計算百分比）
  const calculateTotalDuration = (stages) => {
    return stages.reduce((total, stage) => {
      const duration = stage.duration || 0
      return total + duration
    }, 0)
  }

  const totalDuration = calculateTotalDuration(processingStages.stages)

  // 格式化時間顯示
  const formatDuration = (seconds) => {
    if (!seconds || seconds === 0) return '0.00 秒'
    if (seconds < 0.001) {
      return `${(seconds * 1000).toFixed(2)}ms`
    }
    return `${seconds.toFixed(2)} 秒`
  }

  // 計算百分比
  const calculatePercentage = (duration, total) => {
    if (!total || total === 0) return 0
    return ((duration / total) * 100).toFixed(1)
  }

  // 遞歸渲染子步驟
  const renderSubStages = (subStages, parentDuration, level = 0) => {
    if (!subStages || !Array.isArray(subStages) || subStages.length === 0) return null

    return (
      <Box sx={{ pl: level * 1.5, mt: 0.5 }}>
        {subStages.map((subStage, index) => {
          const isLast = index === subStages.length - 1
          const subDuration = subStage.duration || 0
          const percentage = parentDuration > 0 ? calculatePercentage(subDuration, parentDuration) : 0
          const hasSubStages = subStage.sub_stages && Array.isArray(subStage.sub_stages) && subStage.sub_stages.length > 0

          return (
            <Box key={subStage.name || index} sx={{ mb: 0.5 }}>
              <Typography 
                variant="caption" 
                color="text.secondary"
                sx={{ 
                  fontSize: '0.75rem',
                  display: 'block',
                  lineHeight: 1.6,
                  fontFamily: 'monospace'
                }}
              >
                {(isLast ? '└─' : '├─')} {subStage.label}
                {subDuration > 0 && (
                  <span style={{ marginLeft: '8px', fontWeight: 'normal', fontFamily: 'inherit' }}>
                    {formatDuration(subDuration)}
                    {percentage > 0 && ` (${percentage}%)`}
                  </span>
                )}
              </Typography>
              {/* 遞歸渲染子步驟的子步驟 */}
              {hasSubStages && renderSubStages(subStage.sub_stages, subDuration, level + 1)}
            </Box>
          )
        })}
      </Box>
    )
  }

  const getStepIcon = (stage) => {
    if (stage.status === 'completed') {
      return <CheckCircleIcon color="success" />
    } else if (stage.status === 'in_progress') {
      return <HourglassEmptyIcon color="primary" />
    } else if (stage.status === 'failed') {
      return <ErrorIcon color="error" />
    } else {
      return <RadioButtonUncheckedIcon color="disabled" />
    }
  }

  const getStepColor = (stage) => {
    if (stage.status === 'completed') {
      return 'success'
    } else if (stage.status === 'in_progress') {
      return 'primary'
    } else if (stage.status === 'failed') {
      return 'error'
    } else {
      return 'disabled'
    }
  }

  const activeStep = processingStages.stages.findIndex(
    (stage) => stage.status === 'in_progress'
  )

  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Typography variant="h6" gutterBottom>
        處理進度
      </Typography>
      <Stepper activeStep={activeStep >= 0 ? activeStep : processingStages.stages.length} orientation="vertical">
        {processingStages.stages.map((stage, index) => {
          const stageDuration = stage.duration || 0
          const stagePercentage = totalDuration > 0 ? calculatePercentage(stageDuration, totalDuration) : 0

          return (
            <Step key={stage.name} completed={stage.status === 'completed'}>
              <StepLabel
                StepIconComponent={() => getStepIcon(stage)}
                error={stage.status === 'failed'}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                  <Typography variant="body2">{stage.label}</Typography>
                  {stageDuration > 0 && (
                    <Typography variant="caption" color="text.secondary">
                      {formatDuration(stageDuration)}
                      {stagePercentage > 0 && ` (${stagePercentage}%)`}
                    </Typography>
                  )}
                  {stage.status === 'in_progress' && (
                    <Chip
                      label="進行中"
                      size="small"
                      color="primary"
                      variant="outlined"
                    />
                  )}
                  {stage.status === 'completed' && (
                    <Chip
                      label="完成"
                      size="small"
                      color="success"
                      variant="outlined"
                    />
                  )}
                  {stage.status === 'failed' && (
                    <Chip
                      label="失敗"
                      size="small"
                      color="error"
                      variant="outlined"
                    />
                  )}
                </Box>
              </StepLabel>
              <StepContent>
                <Box sx={{ mt: 1, mb: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <LinearProgress
                      variant={stage.progress !== null && stage.progress !== undefined ? "determinate" : "indeterminate"}
                      value={stage.progress != null ? Math.max(0, Math.min(100, stage.progress)) : 0}
                      sx={{ flexGrow: 1, height: 6, borderRadius: 3 }}
                      color={getStepColor(stage)}
                    />
                    <Typography variant="caption" color="text.secondary" sx={{ minWidth: 40 }}>
                      {stage.progress != null ? `${Math.max(0, Math.min(100, stage.progress))}%` : '--'}
                    </Typography>
                  </Box>
                  {stage.status === 'in_progress' && (
                    <Typography variant="caption" color="primary">
                      正在執行此階段...
                    </Typography>
                  )}
                  {stage.status === 'completed' && (
                    <Box>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                        此階段已完成
                      </Typography>
                      {/* 顯示子步驟時間詳情 */}
                      {(() => {
                        // 調試：檢查 sub_stages 是否存在
                        const hasSubStages = stage.sub_stages && Array.isArray(stage.sub_stages) && stage.sub_stages.length > 0
                        if (stage.name === 'translation_search') {
                          console.log('DEBUG ProcessingStages:', {
                            stageName: stage.name,
                            hasSubStages,
                            subStagesCount: stage.sub_stages ? stage.sub_stages.length : 0,
                            subStages: stage.sub_stages
                          })
                        }
                        return hasSubStages ? (
                          <Box sx={{ mt: 1, pl: 1, borderLeft: '2px solid', borderColor: 'divider' }}>
                            {renderSubStages(stage.sub_stages, stageDuration, 0)}
                          </Box>
                        ) : null
                      })()}
                    </Box>
                  )}
                  {stage.status === 'pending' && (
                    <Typography variant="caption" color="text.secondary">
                      等待執行...
                    </Typography>
                  )}
                  {stage.status === 'failed' && (
                    <Typography variant="caption" color="error">
                      此階段執行失敗
                    </Typography>
                  )}
                </Box>
              </StepContent>
            </Step>
          )
        })}
      </Stepper>
    </Paper>
  )
}

export default ProcessingStages

