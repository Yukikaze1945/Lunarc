import type { CSSProperties } from 'react'
import type { DLSSSettings } from '../types'
import { layerNRSettings, type LayerNRKey } from '../services/multiScaleSettings'
import { ParameterHelp } from './ParameterHelp'

export function LayerNRControls({settings, index, onChange}: {
  settings: DLSSSettings; index: number; onChange: (key: LayerNRKey, value: number) => void
}) {
  const values = layerNRSettings(settings, index)
  return <div className="nr-layer-controls" aria-label={`第 ${index+1} 层参数`}>
    <div className="style-switch" role="group" aria-label={`第 ${index+1} 层画面风格`}>
      {[{value:0,label:'默认'},{value:2,label:'电影'},{value:1,label:'自然'}].map(item =>
        <button type="button" key={item.value} aria-pressed={values.style===item.value}
          onClick={()=>onChange('style', item.value)}>{item.label}</button>)}
    </div>
    {([['intensity','NR 强度'],['local_tone','局部色调强度'],
      ['local_struct','局部结构强度'],['skin_struct','皮肤结构强度']] as const).map(([key,label]) =>
      <label className="primary-slider" key={key}><span>{label}<ParameterHelp name={key} label={label}/>
        <output>{Number(values[key].toFixed(2))}</output></span>
        <input aria-label={`第 ${index+1} 层${label}`} type="range" min={0} max={2} step={.05}
          value={values[key]} style={{'--range-fill':`${values[key]/2*100}%`} as CSSProperties}
          onChange={e=>onChange(key, Number(e.target.value))}/>
      </label>)}
    <label className="primary-check"><input type="checkbox" aria-label={`第 ${index+1} 层自动遮罩`}
      checked={values.use_auto_mask===1} onChange={e=>onChange('use_auto_mask',e.target.checked?1:0)}/>
      自动遮罩<ParameterHelp name="use_auto_mask" label="自动遮罩"/></label>
  </div>
}
