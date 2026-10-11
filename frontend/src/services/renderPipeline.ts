import type { DLSSSettings, RenderStage } from '../types'

const LEGACY: Record<string, RenderStage[]> = {
  nr: ['nr'], dlaa: ['dlaa'], vsr: ['vsr'], nr_dlaa: ['nr', 'dlaa'], nr_vsr: ['nr', 'vsr'],
}
export function renderStages(settings: DLSSSettings): RenderStage[] {
  const stages = settings.render_pipeline ?? LEGACY[settings.neural_mode ?? 'nr'] ?? ['nr']
  // Saved projects from before VFG joined the list keep their original tail position.
  return settings.frame_generation && settings.frame_generation !== 'off' && !stages.includes('vfg')
    ? [...stages, 'vfg'] : stages
}
export function moveStage(stages: RenderStage[], from: number, to: number): RenderStage[] {
  if (from < 0 || to < 0 || from >= stages.length || to >= stages.length) return stages
  const result = [...stages]
  result.splice(to, 0, result.splice(from, 1)[0])
  return result
}
export function pipelinePatch(stages: RenderStage[], settings?: DLSSSettings): Partial<DLSSSettings> {
  return { render_pipeline: stages, neural_mode: 'nr',
    frame_generation: stages.includes('vfg') ? (settings?.frame_generation === 'vfg_dynamic' ? 'vfg_dynamic' : 'vfg') : 'off',
    ...(stages.some(s => s !== 'nr') ? { export_backend: 'gpu_transfer' as const } : {}),
    ...(!stages.includes('nr') ? { working_scale: 1, reconstruction_mode: 'single' as const,
      layer_scales: [1], layer_weights: [1] } : {}),
    ...(!stages.includes('nr') && !stages.includes('dlaa') ? { guidance_mode: 0 } : {}) }
}

export function vsrOutputSize(settings: DLSSSettings, size: {width:number; height:number}): {width:number; height:number} {
  if (settings.vsr_output_mode === 'scale') {
    const factor = settings.vsr_scale ?? 2
    return {width:Math.floor(size.width * factor / 2 + .5) * 2,
      height:Math.floor(size.height * factor / 2 + .5) * 2}
  }
  const choices = { '720p':[1280,720], '1080p':[1920,1080], '1440p':[2560,1440], '4k':[3840,2160], '8k':[7680,4320] }
  const [width,height] = settings.vsr_target === 'custom'
    ? [settings.vsr_width ?? 1920, settings.vsr_height ?? 1080]
    : choices[settings.vsr_target ?? '1080p']
  return {width,height}
}

export function framegenOutputFps(settings: DLSSSettings, sourceFps: number): number {
  if (settings.frame_generation === 'vfg_dynamic') {
    const value = String(settings.framegen_target_fps ?? 60).split('/')
    return Number(value[0]) / (value.length === 2 ? Number(value[1]) : 1)
  }
  return sourceFps * (settings.frame_generation === 'vfg' ? settings.frame_multiplier ?? 2 : 1)
}

export function nativeOutputBlockers(settings: DLSSSettings, video: {width:number; height:number; fps:number} | null): string[] {
  if (!video) return []
  const reasons: string[] = []
  if (renderStages(settings).includes('vsr')) {
    const result = vsrOutputSize(settings, video)
    if (settings.vsr_output_mode === 'scale' && (!Number.isFinite(settings.vsr_scale ?? 2) || (settings.vsr_scale ?? 2) <= 1 || (settings.vsr_scale ?? 2) > 4))
      reasons.push('VSR 倍率须大于 1 且不超过 4。')
    if ([result.width,result.height].some(n => !Number.isInteger(n) || n < 64 || n > 16384 || n % 2 !== 0))
      reasons.push('VSR 宽高须为 64–16384 范围内的偶数。')
    else if (result.width <= video.width || result.height <= video.height || result.width > video.width * 4 || result.height > video.height * 4)
      reasons.push('VSR 输出须放大原图，单轴倍率不超过 4。')
    else if (Math.abs(video.width / video.height - result.width / result.height) > .001)
      reasons.push('VSR 输出须保持素材宽高比。')
  }
  if (settings.frame_generation === 'vfg_dynamic') {
    const fps = framegenOutputFps(settings, video.fps)
    if (!Number.isFinite(fps) || fps <= video.fps)
      reasons.push('动态插帧的目标帧率须高于素材帧率。')
  }
  return reasons
}
