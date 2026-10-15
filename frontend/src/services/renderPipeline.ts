import type { DLSSSettings, RenderStage, RenderStageKind } from '../types'
import {vsrSameSize} from './vsrModes.ts'

export function stageKind(stage:RenderStage):RenderStageKind {return stage.split(':')[0] as RenderStageKind}
export function stageSettings(settings:DLSSSettings,stage:RenderStage):DLSSSettings {return {...settings,...settings.render_stage_options?.[stage]}}
export function stageName(stage:RenderStage):string {return {nr:'DLSSNR',dlaa:'DLAA',vsr:'RTX VSR',vfg:'NVIDIA VFG',truehdr:'TrueHDR',illustration:'IllustrationJaNai V3',animesharp:'AnimeSharp V4'}[stageKind(stage)]}
export function hasScaleStages(settings:DLSSSettings):boolean {return renderStages(settings).some(s=>['vsr','illustration','animesharp'].includes(stageKind(s)))}

const LEGACY: Record<string, RenderStage[]> = {
  nr: ['nr'], dlaa: ['dlaa'], vsr: ['vsr'], nr_dlaa: ['nr', 'dlaa'], nr_vsr: ['nr', 'vsr'],
}
export function configuredRenderStages(settings: DLSSSettings): RenderStage[] {
  const stages = [...new Set(settings.render_pipeline ?? LEGACY[settings.neural_mode ?? 'nr'] ?? ['nr'])]
  // Saved projects from before VFG joined the list keep their original tail position.
  if (settings.frame_generation && settings.frame_generation !== 'off' && !stages.includes('vfg')) stages.push('vfg')
  if (settings.truehdr_enabled && !stages.includes('truehdr')) stages.push('truehdr')
  return stages
}
export function renderStages(settings: DLSSSettings): RenderStage[] {
  return configuredRenderStages(settings).filter(stage => settings.render_stage_enabled?.[stage] !== false)
}
export function toggleStagePatch(stage: RenderStage, settings: DLSSSettings): Partial<DLSSSettings> {
  const enabled = {...settings.render_stage_enabled, [stage]: settings.render_stage_enabled?.[stage] === false}
  return pipelinePatch(configuredRenderStages(settings), {...settings,render_stage_enabled:enabled})
}
export function moveStage(stages: RenderStage[], from: number, to: number): RenderStage[] {
  if (from < 0 || to < 0 || from >= stages.length || to >= stages.length) return stages
  const result = [...stages]
  result.splice(to, 0, result.splice(from, 1)[0])
  return result
}
export function pipelinePatch(stages: RenderStage[], settings?: DLSSSettings): Partial<DLSSSettings> {
  const enabled = Object.fromEntries(Object.entries(settings?.render_stage_enabled ?? {}).filter(([stage]) => stages.includes(stage as RenderStage)))
  const options=Object.fromEntries(Object.entries(settings?.render_stage_options??{}).filter(([stage])=>stages.includes(stage as RenderStage)))
  const active = stages.filter(stage => enabled[stage] !== false)
  const vfgMode = settings?.frame_generation === 'vfg_dynamic' || (settings?.frame_generation === 'off' && settings.frame_generation_saved_mode === 'vfg_dynamic') ? 'vfg_dynamic' : 'vfg'
  return { render_pipeline: stages, neural_mode: 'nr', render_stage_options:options,
    ...(settings?.render_stage_enabled ? {render_stage_enabled: enabled} : {}),
    truehdr_enabled: active.includes('truehdr'),
    frame_generation: active.includes('vfg') ? vfgMode : 'off',
    ...(stages.includes('vfg') ? {frame_generation_saved_mode:vfgMode} : {}),
    ...(stages.some(s => s !== 'nr') ? { export_backend: 'gpu_transfer' as const } : {}),
    ...(!stages.includes('nr') ? { working_scale: 1, reconstruction_mode: 'single' as const,
      layer_scales: [1], layer_weights: [1] } : {}),
    ...(!stages.includes('nr') && !stages.includes('dlaa') ? { guidance_mode: 0 } : {}) }
}

export function vsrOutputSize(settings:DLSSSettings,size:{width:number;height:number}):{width:number;height:number} {
  let result={...size}
  for(const stage of renderStages(settings)){
    const kind=stageKind(stage)
    if(kind==='vsr')result=singleVsrOutputSize(stageSettings(settings,stage),result)
    else if(kind==='illustration'||kind==='animesharp'){
      const scale=kind==='illustration'?4:2;result={width:result.width*scale,height:result.height*scale}
    }
  }
  return result
}
export function singleVsrOutputSize(settings: DLSSSettings, size: {width:number; height:number}): {width:number; height:number} {
  if (vsrSameSize(settings)) return {...size}
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
  if (settings.render_stage_enabled?.vfg === false) return sourceFps
  if (settings.frame_generation === 'vfg_dynamic') {
    const value = String(settings.framegen_target_fps ?? 60).split('/')
    return Number(value[0]) / (value.length === 2 ? Number(value[1]) : 1)
  }
  return sourceFps * (settings.frame_generation === 'vfg' ? settings.frame_multiplier ?? 2 : 1)
}

export function nativeOutputBlockers(settings: DLSSSettings, video: {width:number; height:number; fps:number} | null): string[] {
  if (!video) return []
  const reasons: string[] = []
  let input={width:video.width,height:video.height}
  for(const stage of renderStages(settings)){
    const kind=stageKind(stage),local=stageSettings(settings,stage)
    if(kind==='illustration'||kind==='animesharp'){
      const scale=kind==='illustration'?4:2;input={width:input.width*scale,height:input.height*scale}
      if(Math.max(input.width,input.height)>16384)reasons.push('图片放大链输出单轴不能超过 16384 像素。')
      continue
    }
    if(kind!=='vsr')continue
    const result=singleVsrOutputSize(local,input)
    const same=vsrSameSize(local)
    if(!same&&local.vsr_output_mode==='scale'&&(!Number.isFinite(local.vsr_scale??2)||(local.vsr_scale??2)<=1||(local.vsr_scale??2)>4))reasons.push('VSR 倍率须大于 1 且不超过 4。')
    if([result.width,result.height].some(n=>!Number.isInteger(n)||n<64||n>16384||n%2!==0))reasons.push('VSR 宽高须为 64–16384 范围内的偶数。')
    else if(!same&&(result.width<=input.width||result.height<=input.height||result.width>input.width*4||result.height>input.height*4))reasons.push('每层 VSR 输出须放大该层输入，单轴倍率不超过 4。')
    else if(Math.abs(input.width/input.height-result.width/result.height)>.001)reasons.push('VSR 输出须保持该层输入的宽高比。')
    input=result
  }
  if (settings.frame_generation === 'vfg_dynamic' && settings.render_stage_enabled?.vfg !== false) {
    const fps = framegenOutputFps(settings, video.fps)
    if (!Number.isFinite(fps) || fps <= video.fps)
      reasons.push('动态插帧的目标帧率须高于素材帧率。')
  }
  return reasons
}
