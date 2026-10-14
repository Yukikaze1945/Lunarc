import type {DLSSSettings} from '../types'

export const VSR_MODES = [
  ['standard','标准超分辨率'], ['denoise','降噪'], ['deblur','去模糊'],
  ['high_bitrate','高码率超分辨率'], ['streaming','流媒体超分辨率'], ['bicubic','双三次缩放'],
] as const
export function vsrSameSize(settings: Pick<DLSSSettings,'vsr_mode'>) {
  return settings.vsr_mode === 'denoise' || settings.vsr_mode === 'deblur'
}
export function vsrModePatch(mode: NonNullable<DLSSSettings['vsr_mode']>, settings: DLSSSettings): Partial<DLSSSettings> {
  return {vsr_mode:mode,...(mode==='streaming' && ![2,4].includes(settings.vsr_quality??2) ? {vsr_quality:2} : {})}
}
export const VSR_QUALITY_LABELS = ['低','中','高','极高']
