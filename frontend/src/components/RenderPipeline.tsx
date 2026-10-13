import { useState } from 'react'
import { ArrowUp, ArrowDown, GripVertical, X, Plus, Minus } from 'lucide-react'
import type { DLSSSettings, RenderStage } from '../types'
import { moveStage, pipelinePatch, renderStages } from '../services/renderPipeline'
import './renderPipeline.css'
import { ParameterHelp } from './ParameterHelp'
import { useStageSort } from './useStageSort'
import { TrueHDRTile } from './TrueHDRTile'

function VsrScale({ value, onChange }: { value: number; onChange: (value: number) => void }) {
  const [custom, setCustom] = useState(value !== 2 && value !== 4)
  const [draft, setDraft] = useState<string | null>(null)
  const step = (direction: number) => {
    const next = Math.round(Math.min(4, Math.max(1.01, value + direction)) * 10000) / 10000
    setDraft(null)
    onChange(next)
  }
  return <>
    <label className="render-full">倍率<select aria-label="RTX VSR 倍率预设"
      value={custom || (value !== 2 && value !== 4) ? 'custom' : String(value)}
      onChange={event => { setDraft(null); setCustom(event.target.value === 'custom'); if (event.target.value !== 'custom') onChange(Number(event.target.value)) }}>
      <option value="2">2×</option><option value="4">4×</option><option value="custom">自定义</option>
    </select></label>
    {(custom || (value !== 2 && value !== 4)) && <label className="render-full">自定义
      <div className="render-scale-input"><input aria-label="RTX VSR 输出倍率" type="number" min={1.01} max={4} step="any"
        value={draft ?? value} onChange={event => {
          setDraft(event.target.value)
          const next = Number(event.target.value)
          if (Number.isFinite(next) && next > 1 && next <= 4) onChange(next)
        }} onBlur={() => setDraft(null)} onKeyDown={event => {
          if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
            event.preventDefault(); step(event.key === 'ArrowUp' ? 1 : -1)
          }
        }}/><span>×</span><div className="render-scale-step">
        <button type="button" aria-label="VSR 倍率增加 1" disabled={value >= 4} onClick={() => step(1)}><ArrowUp size={12}/></button>
        <button type="button" aria-label="VSR 倍率减少 1" disabled={value <= 1.01} onClick={() => step(-1)}><ArrowDown size={12}/></button>
      </div></div>
    </label>}
  </>
}

const NAMES: Record<RenderStage, string> = { nr: 'DLSSNR', dlaa: 'DLAA', vsr: 'RTX VSR', vfg: 'NVIDIA VFG', truehdr: 'TrueHDR' }
export function RenderPipeline({ settings, onChange }: {
  settings: DLSSSettings; onChange: (patch: Partial<DLSSSettings>) => void
}) {
  const stages = renderStages(settings)
  const [open,setOpen]=useState(true)
  const [closed,setClosed]=useState<Partial<Record<RenderStage,boolean>>>({})
  const save = (value: RenderStage[]) => onChange(pipelinePatch(value, settings))
  const sort = useStageSort(stages, save)
  return <section className="render-pipeline" aria-label="渲染顺序">
    <header><h3>渲染顺序</h3><span>从上到下</span><button type="button" aria-label={open?'收起渲染顺序':'展开渲染顺序'} aria-expanded={open} onClick={()=>setOpen(!open)}>{open?<Minus size={15}/>:<Plus size={15}/>}</button></header>
    <div hidden={!open}>
    <ol ref={sort.list} className={sort.active ? 'render-sorting' : ''}>{stages.map((stage, i) => <li key={stage}
      data-stage={stage} ref={node => { if (node) sort.nodes.current.set(stage, node); else sort.nodes.current.delete(stage) }}
      className={`render-card${sort.active === stage ? ' render-dragging' : ''}`} style={sort.style(stage)}>
      <div className="render-stage-head"><button type="button" className="render-grip" aria-label={`拖动排序 ${NAMES[stage]}`} {...sort.handle(stage)}><GripVertical size={16}/></button><span className="render-index">{i + 1}</span><strong>{NAMES[stage]}</strong>
        <button type="button" aria-label={`${NAMES[stage]} 上移`} disabled={i === 0} onClick={() => sort.animateSave(moveStage(stages, i, i - 1))}><ArrowUp size={14}/></button>
        <button type="button" aria-label={`${NAMES[stage]} 下移`} disabled={i === stages.length - 1} onClick={() => sort.animateSave(moveStage(stages, i, i + 1))}><ArrowDown size={14}/></button>
        <button type="button" aria-label={`移除 ${NAMES[stage]}`} disabled={stages.length === 1} onClick={() => save(stages.filter(s => s !== stage))}><X size={14}/></button>
        {stage!=='nr'&&<button type="button" aria-label={`${closed[stage]?'展开':'收起'} ${NAMES[stage]} 参数`} aria-expanded={!closed[stage]} onClick={()=>setClosed(value=>({...value,[stage]:!value[stage]}))}>{closed[stage]?<Plus size={14}/>:<Minus size={14}/>}</button>}
      </div>
      <div hidden={closed[stage]}>
      {stage === 'truehdr' && <TrueHDRTile settings={settings} onChange={onChange}/>}
      {stage === 'dlaa' && <div className="render-stage-options render-stage-single">
        <label>模型<ParameterHelp name="dlaa_preset" label="DLAA 模型预设"/><select aria-label="DLAA 模型预设" value={settings.dlaa_preset ?? 0} onChange={e => onChange({dlaa_preset:Number(e.target.value) as DLSSSettings['dlaa_preset']})}>
          <option value={0}>默认</option><option value={10}>J</option><option value={11}>K</option><option value={12}>L</option><option value={13}>M</option><option value={5}>E · 旧版</option><option value={6}>F · 旧版</option>
        </select></label>
      </div>}
      {stage === 'vsr' && <div className="render-stage-options">
        <label>强度<ParameterHelp name="vsr_quality" label="VSR 强度"/><select aria-label="RTX VSR 档位" value={settings.vsr_quality ?? 2} onChange={e => onChange({vsr_quality:Number(e.target.value)})}>{['低','中','高','极高'].map((label, i) => <option key={i + 1} value={i + 1}>{label}</option>)}</select></label>
        <label>输出<ParameterHelp name="vsr_output" label="VSR 输出"/><select aria-label="RTX VSR 输出模式" value={settings.vsr_output_mode ?? 'resolution'} onChange={e => onChange({vsr_output_mode:e.target.value as 'scale' | 'resolution'})}><option value="scale">按倍率</option><option value="resolution">按分辨率</option></select></label>
        {settings.vsr_output_mode === 'scale'
          ? <VsrScale value={settings.vsr_scale ?? 2} onChange={vsr_scale => onChange({vsr_scale})}/>
          : <><label className="render-full">分辨率<select aria-label="RTX VSR 输出" value={settings.vsr_target ?? '1080p'} onChange={e => onChange({vsr_target:e.target.value as DLSSSettings['vsr_target']})}><option value="720p">720p</option><option value="1080p">1080p</option><option value="1440p">1440p</option><option value="4k">4K</option><option value="8k">8K</option><option value="custom">自定义</option></select></label>
            {settings.vsr_target === 'custom' && <><label>宽<input aria-label="RTX VSR 输出宽度" type="number" min={64} max={16384} step={2} value={settings.vsr_width ?? 1920} onChange={e=>onChange({vsr_width:Number(e.target.value)})}/></label><label>高<input aria-label="RTX VSR 输出高度" type="number" min={64} max={16384} step={2} value={settings.vsr_height ?? 1080} onChange={e=>onChange({vsr_height:Number(e.target.value)})}/></label></>}
          </>}
      </div>}
    {stage === 'vfg' && <div aria-label="NVIDIA VFG 卡片">
    <div className="render-stage-options render-framegen-options">
      <label>模式<select aria-label="导出插帧" value={settings.frame_generation ?? 'off'} onChange={e => onChange({frame_generation:e.target.value as DLSSSettings['frame_generation'], ...(e.target.value !== 'off' ? {export_backend:'gpu_transfer' as const} : {})})}>
        <option value="vfg">NVIDIA VFG（固定）</option><option value="vfg_dynamic">NVIDIA VFG（动态）</option>
      </select></label>
      {settings.frame_generation && settings.frame_generation !== 'off' && <>
        {settings.frame_generation === 'vfg'
          ? <label className="render-full">倍率<ParameterHelp name="frame_multiplier" label="插帧倍率"/><select aria-label="插帧倍率" value={settings.frame_multiplier ?? 2} onChange={e => onChange({frame_multiplier:Number(e.target.value) as DLSSSettings['frame_multiplier']})}>{[2,3,4,5,6,7,8].map(n => <option key={n} value={n}>{n}×{n >= 5 ? ' · 实验性' : ''}</option>)}</select></label>
          : <label className="render-full">帧率<ParameterHelp name="framegen_target_fps" label="目标帧率"/><input aria-label="插帧目标帧率" type="text" inputMode="decimal" value={settings.framegen_target_fps ?? 60} onChange={e=>onChange({framegen_target_fps:e.target.value})}/><span>fps</span></label>}
        <label className="render-full">质量<ParameterHelp name="framegen_quality" label="插帧质量"/><select aria-label="插帧质量" value={settings.framegen_quality ?? 1} onChange={e => onChange({framegen_quality:Number(e.target.value) as 0 | 1 | 2})}><option value={0}>低</option><option value={1}>中</option><option value={2}>高</option></select></label>
        <label className="render-full render-check"><input aria-label="VFG 自动镜头切换检测" type="checkbox" checked={settings.framegen_scene_detection ?? true} onChange={e=>onChange({framegen_scene_detection:e.target.checked})}/>镜头切换检测<ParameterHelp name="framegen_scene_detection" label="镜头切换检测"/></label>
      </>}
    </div></div>}
    </div></li>)}</ol>
    <div className="render-add">{(['dlaa','nr','vsr','vfg','truehdr'] as RenderStage[]).filter(s => !stages.includes(s)).map(stage => <button key={stage} type="button" onClick={() => save([...stages,stage])}><Plus size={13}/>{NAMES[stage]}</button>)}</div>
    </div>
  </section>
}
