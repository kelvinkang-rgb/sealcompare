import React, { useMemo } from 'react'
import { Box, Paper, Table, TableBody, TableCell, TableContainer, TableRow, Typography } from '@mui/material'
import { buildTimingCountRows, buildTimingLeafRows, COUNT_GROUPS, sumCounts, sumSeconds, TIMING_GROUPS } from '../utils/timingModel'

const formatSeconds = (v) => {
  const n = typeof v === 'number' ? v : 0
  return `${n.toFixed(3)} 秒`
}

const formatPercent = (part, total) => {
  const p = typeof part === 'number' ? part : 0
  const t = typeof total === 'number' ? total : 0
  if (!t || t <= 0) return ''
  return ` (${((p / t) * 100).toFixed(1)}%)`
}

const formatCount = (v) => {
  const n = typeof v === 'number' ? v : 0
  // eval 次數應為整數，但後端目前用 float 存；這裡保守顯示 0 位小數
  return `${n.toFixed(0)} 次`
}

export default function TimingDetailsTable({ timing, alignmentMetrics }) {
  // 額外：把右角候選的「純額外耗時」注入到 timing，方便用同一張表顯示（秒）
  const timingWithExtras = useMemo(() => {
    const t = timing && typeof timing === 'object' ? timing : {}
    const m = alignmentMetrics && typeof alignmentMetrics === 'object' ? alignmentMetrics : {}
    const extra = m?.right_angle_extra_overhead_time
    if (typeof extra === 'number' && Number.isFinite(extra) && extra > 0) {
      return { ...t, right_angle_extra_overhead_time: extra }
    }
    return t
  }, [timing, alignmentMetrics])

  const rows = useMemo(() => buildTimingLeafRows(timingWithExtras), [timingWithExtras])
  const overallTotal = useMemo(() => sumSeconds(rows), [rows])
  const countRows = useMemo(() => buildTimingCountRows(timingWithExtras), [timingWithExtras])

  const rowsByGroup = useMemo(() => {
    const m = new Map()
    for (const g of TIMING_GROUPS) m.set(g.id, [])
    for (const r of rows) {
      const arr = m.get(r.group) || []
      arr.push(r)
      m.set(r.group, arr)
    }
    return m
  }, [rows])

  const hasAny = rows.length > 0
  const hasAnyCounts = countRows.length > 0
  if (!hasAny && !hasAnyCounts) {
    return (
      <Typography variant="body2" color="text.secondary">
        無可顯示的拆解（時間只顯示秒；計數以次顯示；父層總時間已隱藏）
      </Typography>
    )
  }

  return (
    <Box>
      <TableContainer component={Paper} variant="outlined">
        <Table size="small">
          <TableBody>
            {TIMING_GROUPS.map((g) => {
              const groupRowsRaw = rowsByGroup.get(g.id) || []
              const groupRows = [...groupRowsRaw].sort((a, b) => {
                const ao = typeof a?.order === 'number' ? a.order : 0
                const bo = typeof b?.order === 'number' ? b.order : 0
                if (ao !== bo) return ao - bo
                const al = String(a?.label ?? '')
                const bl = String(b?.label ?? '')
                return al.localeCompare(bl)
              })
              if (!groupRows || groupRows.length === 0) return null

              const groupTotal = sumSeconds(groupRows)

              return (
                <React.Fragment key={g.id}>
                  <TableRow>
                    <TableCell sx={{ backgroundColor: 'grey.100', fontWeight: 'bold' }}>{g.label}</TableCell>
                    <TableCell align="right" sx={{ backgroundColor: 'grey.100', fontWeight: 'bold' }}>
                      {formatSeconds(groupTotal)}
                      {formatPercent(groupTotal, overallTotal)}
                    </TableCell>
                  </TableRow>
                  {groupRows.map((r) => (
                    <TableRow key={r.key}>
                      <TableCell sx={{ color: 'text.secondary', fontFamily: 'monospace' }}>{r.label}</TableCell>
                      <TableCell align="right" sx={{ color: 'text.secondary', fontFamily: 'monospace' }}>
                        {formatSeconds(r.seconds)}
                        {formatPercent(r.seconds, groupTotal)}
                      </TableCell>
                    </TableRow>
                  ))}
                </React.Fragment>
              )
            })}

            {hasAnyCounts && (
              <>
                {COUNT_GROUPS.map((g) => {
                  const groupRows = [...countRows].sort((a, b) => {
                    const ao = typeof a?.order === 'number' ? a.order : 0
                    const bo = typeof b?.order === 'number' ? b.order : 0
                    if (ao !== bo) return ao - bo
                    const al = String(a?.label ?? '')
                    const bl = String(b?.label ?? '')
                    return al.localeCompare(bl)
                  })
                  const groupTotal = sumCounts(groupRows)
                  return (
                    <React.Fragment key={g.id}>
                      <TableRow>
                        <TableCell sx={{ backgroundColor: 'grey.100', fontWeight: 'bold' }}>{g.label}</TableCell>
                        <TableCell align="right" sx={{ backgroundColor: 'grey.100', fontWeight: 'bold' }}>
                          {formatCount(groupTotal)}
                        </TableCell>
                      </TableRow>
                      {groupRows.map((r) => (
                        <TableRow key={r.key}>
                          <TableCell sx={{ color: 'text.secondary', fontFamily: 'monospace' }}>{r.label}</TableCell>
                          <TableCell align="right" sx={{ color: 'text.secondary', fontFamily: 'monospace' }}>
                            {formatCount(r.count)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </React.Fragment>
                  )
                })}
              </>
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  )
}


