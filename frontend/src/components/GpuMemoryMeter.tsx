import {ui} from '../i18n/locale'
import {memoryLevel} from '../services/processingFeedback'
export function GpuMemoryMeter({used,total,label='显存',unit='GB',compact=false}:{used?:number;total?:number;label?:string;unit?:string;compact?:boolean}){
  const value=memoryLevel(used,total)
  return <div className={`gpu-memory-meter${compact?' gpu-memory-compact':''}`} data-level={value?.level||'unknown'}
    title={compact?ui('整张显卡显存占用，包含其他程序；约每 5 秒采样。'):undefined}>
    <span className="gpu-memory-label">{ui(label)} {value?`${used!.toFixed(1)} / ${total!.toFixed(1)} ${unit}`:'—'}
      {value&&<span className="gpu-memory-percent">{Math.round(value.percent)}%</span>}</span>
    <div className="gpu-memory-track" role="meter" aria-label={ui(label)} aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={value?Math.min(100,Math.round(value.percent)):undefined}
      aria-valuetext={ui(!value?'暂无显存数据':value.level==='critical'?'显存接近满载':value.level==='warning'?'显存用量较高':'显存余量充足')}>
      <div style={{width:`${value?.fill||0}%`}}/>
    </div>
  </div>
}
