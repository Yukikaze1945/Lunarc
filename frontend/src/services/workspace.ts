import type { DLSSSettings } from "../types"
import { DEFAULT_POST, DEFAULT_TEMPORAL } from "./qualityControls"
export const DEFAULT_SETTINGS: DLSSSettings = {
  neural_mode: 'nr',
  vsr_quality: 2,
  vsr_target: '1080p', vsr_output_mode: 'resolution', vsr_scale: 2, vsr_width:1920, vsr_height:1080,
  frame_generation: 'off', frame_multiplier: 2, framegen_quality: 1,
  framegen_target_fps: 60, framegen_scene_detection: true,
  vsr_runtime_path: '',
  dlss_dll_path: '',
  preset: 1,
  style: 0,
  intensity: 1.0,
  local_tone: 1.0,
  local_struct: 1.0,
  skin_struct: 1.0,
  use_auto_mask: 1,
  ui_correction: 0,
  guidance_mode: 1,
  motion_scale_x: 1.0,
  motion_scale_y: 1.0,
  working_scale: 1.0,
  residual_pyramid: true,
  residual_strength: 1.0,
  flow_engine: 'nvof',
  flow_quality: '1F',
  export_backend: 'auto',
  reconstruction_mode: 'single',
  layer_scales: [1.0],
  layer_weights: [1.0],
  fusion_mode: 'catmull_baseline',
  fusion_beta: 0.1,
  fusion_max_iterations: 128,
  fusion_relative_tolerance: 0.001,
  // The service answers GET /api/settings with its own defaults and this object
  // is only what shows before that reply lands, so the two have to agree: the
  // project GPU host is the only DLSS host this build has.
  dlss_backend: 'gpu_flow_d3d12',
  // Experimental HDR10 export stays off until the user opts in, and the
  // paper-white value here is the backend's own fixed default (8.0 scRGB =
  // 640 cd/m²). The backend is authoritative: it refuses a request it cannot
  // serve rather than degrading it.
  experimental_hdr_scrgb: false,
  hdr_paper_white_scrgb: 8.0,
  post_controls: DEFAULT_POST,
  temporal_controls: DEFAULT_TEMPORAL,
}

export function completeSettings(raw: Partial<DLSSSettings>): DLSSSettings {
  return {...DEFAULT_SETTINGS, ...raw,
    // Retire the unsupported style from saved workspace and imported presets.
    style: raw.style === 3 ? DEFAULT_SETTINGS.style : (raw.style ?? DEFAULT_SETTINGS.style),
    post_controls: {...DEFAULT_POST, ...raw.post_controls},
    temporal_controls: {...DEFAULT_TEMPORAL, ...raw.temporal_controls}}
}
export const WORKSPACE_KEY = 'lunarc.workspace.v1'
export interface WorkspaceDraft { version: 1; settings: DLSSSettings; recent: string[] }
export function readWorkspace(): WorkspaceDraft | null {
  try { const d = JSON.parse(localStorage.getItem(WORKSPACE_KEY) || 'null');
    if (d?.version !== 1 || !d.settings || typeof d.settings !== 'object' || Array.isArray(d.settings)) return null
    return {version: 1, settings: completeSettings(d.settings),
      recent: Array.isArray(d.recent) ? d.recent.filter((p: unknown) => typeof p === 'string').slice(0,8) : []}
  } catch { return null }
}
export function storeWorkspace(settings: DLSSSettings, recent: string[]) {
  localStorage.setItem(WORKSPACE_KEY, JSON.stringify({version: 1, settings, recent}))
}
