import test from 'node:test'
import assert from 'node:assert/strict'
import { buildTimingCountRows, buildTimingLeafRows, sumCounts, sumSeconds } from './timingModel.js'

test('buildTimingLeafRows: handles empty input', () => {
  const rows = buildTimingLeafRows(null)
  assert.deepEqual(rows, [])
})

test('buildTimingLeafRows: filters parent keys and keeps leaf numbers', () => {
  const timing = {
    total: 9.9, // parent
    remove_bg_image1: 1.23, // parent
    load_images: 0.1,
    create_heatmap: 0.2,
    alignment_stages: {
      remove_background_total: 0.5, // parent
      step1_convert_to_gray: 0.01,
      stage1_translation_coarse: 0.02,
      some_unknown_leaf: 0.03,
      stage3_translation_rescue_evals: 2131, // not seconds
      post_fine_rescue_evals: 1251, // not seconds
    },
    image1_background_stages: {
      remove_background_total: 0.4, // parent
      step2_detect_bg_color: 0.04,
      another_unknown_leaf: 0.05,
    },
    weird_object: { a: 1 }, // not leaf
    weird_string: 'x', // not leaf
  }

  const rows = buildTimingLeafRows(timing)
  const keys = new Set(rows.map(r => r.key))

  assert.equal(keys.has('total'), false)
  assert.equal(keys.has('remove_bg_image1'), false)

  // task leaf
  assert.equal(keys.has('load_images'), true)
  assert.equal(keys.has('create_heatmap'), true)

  // image1 leaf
  assert.equal(keys.has('image1:step2_detect_bg_color'), true)
  assert.equal(keys.has('image1:another_unknown_leaf'), true)

  // image2 (alignment_stages) leaf
  assert.equal(keys.has('image2:step1_convert_to_gray'), true)
  assert.equal(keys.has('align:stage1_translation_coarse'), true)
  assert.equal(keys.has('align:some_unknown_leaf'), true)
  assert.equal(keys.has('align:stage3_translation_rescue_evals'), false)
  assert.equal(keys.has('align:post_fine_rescue_evals'), false)

  // no parent remove_background_total
  assert.equal(keys.has('align:remove_background_total'), false)
  assert.equal(keys.has('image1:remove_background_total'), false)
})

test('buildTimingCountRows: collects *_evals as counts (次)', () => {
  const timing = {
    alignment_stages: {
      stage3_translation_rescue_evals: 2131,
      post_fine_rescue_evals: 1251,
      stage3_translation_rescue_total: 7.0, // seconds, not counts
    },
  }
  const rows = buildTimingCountRows(timing)
  const keys = new Set(rows.map(r => r.key))
  assert.equal(keys.has('align:stage3_translation_rescue_evals'), true)
  assert.equal(keys.has('align:post_fine_rescue_evals'), true)
  assert.equal(keys.has('align:stage3_translation_rescue_total'), false)
  assert.equal(rows.find(r => r.key === 'align:stage3_translation_rescue_evals')?.unit, '次')
  assert.equal(sumCounts(rows), 3382)
})

test('sumSeconds: sums only numeric seconds', () => {
  const rows = [
    { seconds: 0.1 },
    { seconds: 0.2 },
    { seconds: null },
    { seconds: NaN },
  ]
  assert.equal(sumSeconds(rows), 0.30000000000000004)
})


