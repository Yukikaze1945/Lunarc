import type { PostControls, TemporalControls } from '../types'
export const DEFAULT_POST: PostControls = {
  total_gain: 1, lightness_gain: 1, chroma_gain: 1, low_frequency_gain: 1,
  high_frequency_gain: 1, hue_protection: 0, dark_protection: 0,
  highlight_protection: 0, soft_compression: 0, roi_kind: 0,
  roi_bounds: [0, 0, 1, 1], roi_feather: .02, debug_mode: 0,
}
export const DEFAULT_TEMPORAL: TemporalControls = {
  scene_detection: true, mode: 'off', history_weight: .6,
}
export function hasExportControls(settings: {post_controls?: PostControls; temporal_controls?: TemporalControls}) {
  const p = {...DEFAULT_POST, ...settings.post_controls}
  const t = {...DEFAULT_TEMPORAL, ...settings.temporal_controls}
  return Object.keys(DEFAULT_POST).some(k => k !== 'roi_bounds' && k !== 'roi_feather' &&
    p[k as keyof PostControls] !== DEFAULT_POST[k as keyof PostControls]) ||
    t.scene_detection || t.mode !== 'off'
}
