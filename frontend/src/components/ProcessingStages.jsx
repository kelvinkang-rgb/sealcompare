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
        {processingStages.stages.map((stage, index) => (
          <Step key={stage.name} completed={stage.status === 'completed'}>
            <StepLabel
              StepIconComponent={() => getStepIcon(stage)}
              error={stage.status === 'failed'}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="body2">{stage.label}</Typography>
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
                    variant="determinate"
                    value={stage.progress}
                    sx={{ flexGrow: 1 }}
                    color={getStepColor(stage)}
                  />
                  <Typography variant="caption" color="text.secondary" sx={{ minWidth: 40 }}>
                    {stage.progress}%
                  </Typography>
                </Box>
                {stage.status === 'in_progress' && (
                  <Typography variant="caption" color="primary">
                    正在執行此階段...
                  </Typography>
                )}
                {stage.status === 'completed' && (
                  <Typography variant="caption" color="text.secondary">
                    此階段已完成
                  </Typography>
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
        ))}
      </Stepper>
    </Paper>
  )
}

export default ProcessingStages

