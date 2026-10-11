import type { DLSSSettings, ReconstructionMode } from '../types'

const percent = (scale: number) => Math.round(scale * 100)
const toScale = (value: number) => value / 100

export function activeLayerCount(mode: ReconstructionMode): number {
  return mode === 'triple' ? 3 : mode === 'dual' ? 2 : 1
}

export function chooseReconstructionMode(
  settings: DLSSSettings, mode: ReconstructionMode,
): Partial<DLSSSettings> {
  const first = percent(settings.working_scale)
  if (mode === 'single') {
    return { reconstruction_mode: 'single', layer_scales: [toScale(first)],
      layer_weights: [1], fusion_mode: 'catmull_baseline' }
  }
  const candidates = [first,
    ...(settings.layer_scales || []).slice(1).map(percent),
    75, 50, 100, 67, 25]
  const unique: number[] = []
  for (const value of candidates) {
    if (value >= 25 && value <= 100 && !unique.includes(value)) unique.push(value)
    if (unique.length === activeLayerCount(mode)) break
  }
  return {
    reconstruction_mode: mode,
    export_backend: 'gpu_transfer',
    layer_scales: unique.map(toScale),
    layer_weights: unique.map((_, index) => settings.layer_weights?.[index] || 1),
  }
}

export function chooseWorkingScale(
  settings: DLSSSettings, scale: number,
): Pick<DLSSSettings, 'working_scale' | 'layer_scales'> {
  const next = percent(scale)
  const previous = percent(settings.working_scale)
  const count = activeLayerCount(settings.reconstruction_mode || 'single')
  const scales = [previous, ...(settings.layer_scales || []).slice(1).map(percent)]
    .slice(0, count)
  const duplicate = scales.indexOf(next, 1)
  if (duplicate >= 1) scales[duplicate] = previous
  scales[0] = next
  return { working_scale: toScale(next), layer_scales: scales.map(toScale) }
}

export function chooseLayerScale(settings: DLSSSettings, index: number, scale: number): number[] {
  const count = activeLayerCount(settings.reconstruction_mode || 'single')
  const next = percent(scale)
  const scales = [percent(settings.working_scale),
    ...(settings.layer_scales || []).slice(1).map(percent)].slice(0, count)
  if (index < 1 || index >= count || next < 25 || next > 100 ||
      scales.some((value, at) => at !== index && value === next)) {
    throw new RangeError('每层必须使用不同的 25–100% 整数比例')
  }
  scales[index] = next
  return scales.map(toScale)
}
