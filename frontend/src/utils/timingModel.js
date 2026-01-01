const BG_STEP_DEFS = [
  ['step1_convert_to_gray', '轉換灰度'],
  ['step1b_shading_correction', '陰影/摺痕抑制（光照校正）'],
  ['step2_detect_bg_color', '偵測背景色'],
  ['step3_otsu_threshold', 'OTSU二值化'],
  ['step3b_remove_thin_lines', '線條型雜訊抑制'],
  ['step4_combine_masks', '結合遮罩'],
  ['step6_contour_detection', '輪廓偵測'],
  ['step6_fallback_red_seal_segmentation', 'fallback：紅章分割（無輪廓）'],
  ['step7_calculate_bbox', '計算邊界框'],
  ['step7_fallback_red_seal_segmentation', 'fallback：紅章分割（輪廓太小）'],
  ['step8_crop_image', '裁切圖像'],
  ['step9_remove_bg_final', '最終移除背景'],
]

const ALIGN_STAGE_DEFS = [
  ['stage12_joint_coarse_total', '階段1+2：joint 粗搜尋（總計）'],
  ['stage1_translation_coarse', '階段1：平移粗調'],
  ['stage2_rotation_coarse', '階段2：旋轉粗調'],
  ['stage3_translation_fine', '階段3：平移細調'],
  ['stage4_total', '階段4：旋轉細調與平移細調（總計）'],
  ['stage4_rotation_fine', '階段4：旋轉細調'],
  ['stage4_translation_fine', '階段4：平移細調'],
  ['stage5_global_verification', '階段5：全局驗證'],
]

const TASK_LEAF_DEFS = [
  ['load_images', '載入圖像'],
  ['save_aligned_images', '保存對齊圖像'],
  ['similarity_calculation', '相似度計算'],
  ['save_corrected_images', '保存校正圖像'],
  ['create_overlay', '生成疊圖'],
  ['calculate_mask_stats', '計算Mask統計'],
  ['create_heatmap', '生成熱力圖'],
]

const PARENT_KEYS = new Set([
  // 任務級父層
  'total',
  'remove_bg_image1',
  'remove_bg_align_image2',
  // 巢狀 dict
  'alignment_stages',
  'image1_background_stages',
  // 去背景父層（內部 sum）
  'remove_background_total',
  'remove_background',
])

function isFiniteNumber(v) {
  return typeof v === 'number' && Number.isFinite(v)
}

function isSecondsLeafKey(key) {
  // 保證「時間詳情」單位一致：只收集秒，不混入計數/次數
  // 目前已知非秒的 key：*_evals（代表評估次數）
  if (typeof key !== 'string') return false
  if (key.endsWith('_evals')) return false
  return true
}

function classifyAlignmentExtraLeaf(key) {
  // 將常見的 alignment rescue / scale 類 key 從 other 拉回 alignment_steps，並給友善 label + order
  if (typeof key !== 'string') return null
  if (!isSecondsLeafKey(key)) return null

  // stage12_scale_{scale}
  const mScale = key.match(/^stage12_scale_(.+)$/)
  if (mScale) {
    return {
      group: 'alignment_steps',
      label: `階段1+2：scale=${mScale[1]}`,
      order: 50,
    }
  }

  // stage3_translation_rescue_total / step_{n}
  if (key === 'stage3_translation_rescue_total') {
    return { group: 'alignment_steps', label: '平移救援（總計）', order: 60 }
  }
  const mRescueStep = key.match(/^stage3_translation_rescue_step_(\d+)$/)
  if (mRescueStep) {
    const n = Number(mRescueStep[1])
    return { group: 'alignment_steps', label: `平移救援 step=${mRescueStep[1]}`, order: 61 + (Number.isFinite(n) ? n : 0) }
  }

  // post fine rescue totals
  if (key === 'post_fine_rescue_total') {
    return { group: 'alignment_steps', label: '後救援（總計）', order: 80 }
  }
  if (key === 'post_fine_rescue_guardrail_total') {
    return { group: 'alignment_steps', label: '後救援保護欄（總計）', order: 81 }
  }

  return null
}

function makeLabelMap(defs) {
  const m = new Map()
  for (const [k, label] of defs) m.set(k, label)
  return m
}

const BG_LABELS = makeLabelMap(BG_STEP_DEFS)
const ALIGN_LABELS = makeLabelMap(ALIGN_STAGE_DEFS)
const TASK_LABELS = makeLabelMap(TASK_LEAF_DEFS)

export function buildTimingLeafRows(timing) {
  const t = timing && typeof timing === 'object' ? timing : {}

  const rows = []

  // 1) 任務級：只取 leaf keys（不取 total / remove_bg_*）
  for (let i = 0; i < TASK_LEAF_DEFS.length; i++) {
    const [k, label] = TASK_LEAF_DEFS[i]
    const v = t?.[k]
    if (isFiniteNumber(v)) rows.push({ key: k, label, seconds: v, group: 'task', order: i })
  }

  // 2) 圖像1去背景 steps：從 image1_background_stages 取 leaf（不顯示 remove_background_total）
  const img1 = t?.image1_background_stages
  if (img1 && typeof img1 === 'object') {
    for (let i = 0; i < BG_STEP_DEFS.length; i++) {
      const [k, label] = BG_STEP_DEFS[i]
      const v = img1?.[k]
      if (isFiniteNumber(v)) rows.push({ key: `image1:${k}`, label, seconds: v, group: 'image1_remove_bg_steps', order: i })
    }
    // 其他未識別 leaf key
    for (const [k, v] of Object.entries(img1)) {
      if (PARENT_KEYS.has(k)) continue
      if (BG_LABELS.has(k)) continue
      if (!isSecondsLeafKey(k)) continue
      if (isFiniteNumber(v)) rows.push({ key: `image1:${k}`, label: k, seconds: v, group: 'other', order: 10000 })
    }
  }

  // 3) 圖像2去背景 steps + 對齊：從 alignment_stages 取 leaf
  const align = t?.alignment_stages
  if (align && typeof align === 'object') {
    // 3a) 去背景 steps
    for (let i = 0; i < BG_STEP_DEFS.length; i++) {
      const [k, label] = BG_STEP_DEFS[i]
      const v = align?.[k]
      if (isFiniteNumber(v)) rows.push({ key: `image2:${k}`, label, seconds: v, group: 'image2_remove_bg_steps', order: i })
    }

    // 3b) 對齊 stages
    for (let i = 0; i < ALIGN_STAGE_DEFS.length; i++) {
      const [k, label] = ALIGN_STAGE_DEFS[i]
      const v = align?.[k]
      if (isFiniteNumber(v)) rows.push({ key: `align:${k}`, label, seconds: v, group: 'alignment_steps', order: i })
    }

    // 其他未識別 leaf key
    for (const [k, v] of Object.entries(align)) {
      if (PARENT_KEYS.has(k)) continue
      if (BG_LABELS.has(k)) continue
      if (ALIGN_LABELS.has(k)) continue
      const extra = classifyAlignmentExtraLeaf(k)
      if (extra) {
        if (isFiniteNumber(v)) rows.push({ key: `align:${k}`, label: extra.label, seconds: v, group: extra.group, order: extra.order })
        continue
      }
      if (!isSecondsLeafKey(k)) continue
      if (isFiniteNumber(v)) rows.push({ key: `align:${k}`, label: k, seconds: v, group: 'other', order: 10000 })
    }
  }

  // 4) timing 其他 leaf key（排除父層與巢狀 dict）
  for (const [k, v] of Object.entries(t)) {
    if (PARENT_KEYS.has(k)) continue
    if (k === 'alignment_stages' || k === 'image1_background_stages') continue
    if (TASK_LABELS.has(k)) continue
    if (!isSecondsLeafKey(k)) continue
    if (isFiniteNumber(v)) rows.push({ key: k, label: k, seconds: v, group: 'other', order: 20000 })
  }

  return rows
}

export function sumSeconds(rows) {
  return (rows || []).reduce((acc, r) => acc + (isFiniteNumber(r?.seconds) ? r.seconds : 0), 0)
}

function isCountLeafKey(key) {
  if (typeof key !== 'string') return false
  return key.endsWith('_evals')
}

function classifyCountKey(key) {
  if (key === 'stage3_translation_rescue_evals') return { label: '平移救援 eval 次數', order: 10 }
  if (key === 'post_fine_rescue_evals') return { label: '後救援 eval 次數', order: 20 }
  if (key === 'angle_sign_micro_rescue_evals') return { label: '角度符號微救援 eval 次數', order: 30 }
  return { label: key, order: 10000 }
}

export function buildTimingCountRows(timing) {
  const t = timing && typeof timing === 'object' ? timing : {}
  const rows = []

  const pushCount = (prefix, key, v) => {
    if (!isCountLeafKey(key)) return
    if (!isFiniteNumber(v)) return
    const meta = classifyCountKey(key)
    rows.push({
      key: `${prefix}${key}`,
      label: meta.label,
      count: v,
      unit: '次',
      group: 'counts',
      order: meta.order,
    })
  }

  // alignment_stages is the main source of eval counts
  const align = t?.alignment_stages
  if (align && typeof align === 'object') {
    for (const [k, v] of Object.entries(align)) pushCount('align:', k, v)
  }

  // also allow top-level counts (if ever added)
  for (const [k, v] of Object.entries(t)) {
    if (k === 'alignment_stages' || k === 'image1_background_stages') continue
    if (PARENT_KEYS.has(k)) continue
    pushCount('', k, v)
  }

  return rows
}

export function sumCounts(rows) {
  return (rows || []).reduce((acc, r) => acc + (isFiniteNumber(r?.count) ? r.count : 0), 0)
}

export const TIMING_GROUPS = [
  { id: 'task', label: '任務步驟（葉節點）' },
  { id: 'image1_remove_bg_steps', label: '圖像1去背景（葉節點）' },
  { id: 'image2_remove_bg_steps', label: '圖像2去背景（葉節點）' },
  { id: 'alignment_steps', label: '對齊（葉節點）' },
  { id: 'other', label: '其他（葉節點）' },
]

export const COUNT_GROUPS = [{ id: 'counts', label: '計數（次）' }]


