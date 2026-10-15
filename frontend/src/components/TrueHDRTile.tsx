// localized-render-sinks
import {ui} from '../i18n/locale'
import type { CSSProperties } from 'react'
import type { DLSSSettings } from '../types'

export function TrueHDRTile({settings,onChange}: {
  settings: DLSSSettings; onChange: (patch: Partial<DLSSSettings>) => void
}) {
  return <div className="truehdr-tile" aria-label={ui("TrueHDR 参数")}>
    {ui(([
      ['truehdr_contrast','对比度',0,200,100],
      ['truehdr_saturation','饱和度',0,200,100],
      ['truehdr_middle_gray','中灰亮度',10,100,50],
      ['truehdr_peak_nits','目标峰值',400,2000,650],
    ] as const).map(([key,label,min,max,initial]) => {
      const value=settings[key]??initial
      return <label className="primary-slider" key={key}><span>{ui(label)}<output>{ui(value)}{ui(key==='truehdr_peak_nits'?' nit':'')}</output></span>
        <input aria-label={ui(`TrueHDR ${label}`)} type="range" min={min} max={max} step={1} value={value}
          style={{'--range-fill':`${(value-min)/(max-min)*100}%`} as CSSProperties}
          onChange={e=>onChange({[key]:Number(e.target.value)})}/></label>
    }))}
    <label className="primary-check"><input type="checkbox" aria-label={ui("TrueHDR AI 去色带")}
      checked={settings.truehdr_debanding??true} onChange={e=>onChange({truehdr_debanding:e.target.checked})}/>{ui("AI 去色带")}</label>
  </div>
}
