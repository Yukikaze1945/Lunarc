/**
 * Pure UI-side export-control selection semantics. These tests exercise the
 * defaults and settings predicate without invoking the native GPU path.
 *
 *   node --test tests/node/quality_controls.test.ts
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'

import { DEFAULT_POST, DEFAULT_TEMPORAL, hasExportControls } from '../../frontend/src/services/qualityControls.ts'

test('strict post and temporal defaults are neutral and preserve their nested ROI bounds', () => {
  assert.deepEqual(DEFAULT_POST, {
    total_gain: 1, lightness_gain: 1, chroma_gain: 1,
    low_frequency_gain: 1, high_frequency_gain: 1,
    hue_protection: 0, dark_protection: 0, highlight_protection: 0,
    soft_compression: 0, roi_kind: 0, roi_bounds: [0, 0, 1, 1],
    roi_feather: 0.02, debug_mode: 0,
  })
  assert.deepEqual(DEFAULT_TEMPORAL, { scene_detection: true, mode: 'off', history_weight: 0.6 })
  assert.equal(hasExportControls({}), true)
  assert.equal(hasExportControls({ post_controls: DEFAULT_POST, temporal_controls: DEFAULT_TEMPORAL }), true)
  assert.equal(hasExportControls({temporal_controls: {...DEFAULT_TEMPORAL, scene_detection: false}}), false)
})

test('only enabled post controls select the export processing path', () => {
  const neutral = { ...DEFAULT_POST, roi_bounds: [...DEFAULT_POST.roi_bounds] as [number, number, number, number] }
  // Rectangle coordinates and feather are dormant while ROI is disabled.
  neutral.roi_bounds = [0.1, 0.2, 0.8, 0.9]
  neutral.roi_feather = 0.1
  assert.equal(hasExportControls({ post_controls: neutral, temporal_controls: {...DEFAULT_TEMPORAL, scene_detection: false} }), false)

  const activeCases = [
    { ...DEFAULT_POST, total_gain: 1.1 },
    { ...DEFAULT_POST, lightness_gain: 0.9 },
    { ...DEFAULT_POST, chroma_gain: 1.1 },
    { ...DEFAULT_POST, low_frequency_gain: 0.9 },
    { ...DEFAULT_POST, high_frequency_gain: 1.1 },
    { ...DEFAULT_POST, hue_protection: 0.1 },
    { ...DEFAULT_POST, dark_protection: 0.1 },
    { ...DEFAULT_POST, highlight_protection: 0.1 },
    { ...DEFAULT_POST, soft_compression: 0.1 },
    { ...DEFAULT_POST, roi_kind: 1 as const },
    { ...DEFAULT_POST, debug_mode: 1 as const },
    { ...DEFAULT_POST, debug_mode: 2 as const },
    { ...DEFAULT_POST, debug_mode: 3 as const },
  ]
  for (const post_controls of activeCases) {
    assert.equal(hasExportControls({ post_controls }), true, JSON.stringify(post_controls))
  }
})

test('scene detection and static low-frequency mode select GPU export; dormant history weight does not', () => {
  assert.equal(hasExportControls({ temporal_controls: { ...DEFAULT_TEMPORAL, scene_detection: false, history_weight: 0.8 } }), false)
  assert.equal(hasExportControls({ temporal_controls: { ...DEFAULT_TEMPORAL, scene_detection: true } }), true)
  assert.equal(hasExportControls({ temporal_controls: { ...DEFAULT_TEMPORAL, mode: 'static_low_frequency' } }), true)
})

test('reading controls preserves nested bounds and leaves the supplied object unchanged', () => {
  const post_controls = { ...DEFAULT_POST, roi_kind: 2 as const, roi_bounds: [0.15, 0.2, 0.75, 0.8] as [number, number, number, number], roi_feather: 0.04 }
  const before = structuredClone(post_controls)
  assert.equal(hasExportControls({ post_controls }), true)
  assert.deepEqual(post_controls, before)
  assert.deepEqual(post_controls.roi_bounds, [0.15, 0.2, 0.75, 0.8])
})
