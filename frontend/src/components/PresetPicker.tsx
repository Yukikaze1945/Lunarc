// localized-render-sinks
import {ui} from '../i18n/locale'
import {useState,useSyncExternalStore} from 'react'
import {Save} from 'lucide-react'
import type {DLSSSettings} from '../types'
import {presetSnapshot,subscribePresets,savePreset,DEFAULT_PRESET,samePreset} from '../services/presets'

export function PresetPicker({settings,onChange,onStatus}:{settings:DLSSSettings;
  onChange:(patch:Partial<DLSSSettings>)=>void;onStatus?:(message:string)=>void}) {
  const presets=useSyncExternalStore(subscribePresets,presetSnapshot)
  const [selected,setSelected]=useState(''),[saving,setSaving]=useState(false),[name,setName]=useState('')
  const active=selected==='__default__'?DEFAULT_PRESET:presets.find(p=>p.name===selected)?.settings
  const current=active&&samePreset(settings,active)?selected:''
  const apply=(value:string)=>{
    setSelected(value)
    const next=value==='__default__'?DEFAULT_PRESET:presets.find(p=>p.name===value)?.settings
    if(next)onChange(next)
  }
  return <div className="workbench-presets">
    <label>{ui("预设")}<select aria-label={ui("选择工作台预设")} value={current} onChange={e=>apply(e.target.value)}>
      <option value="">{ui("当前参数")}</option><option value="__default__">{ui("默认")}</option>
      {ui(presets.map(p=><option key={p.name} value={p.name}>{p.name}</option>))}</select></label>
    <button type="button" className="button quiet" onClick={()=>{setSaving(!saving);setName(selected==='__default__'?'':selected)}}><Save size={14}/>{ui("保存预设")}</button>
    {ui(saving&&<form className="preset-save-form" onSubmit={e=>{e.preventDefault();try{savePreset(name,settings);setSelected(name.trim());setSaving(false);onStatus?.('预设已保存')}catch(error:any){onStatus?.(error.message)}}}>
      <input aria-label={ui("保存预设名称")} autoFocus placeholder={ui("预设名称")} maxLength={80} value={name} onChange={e=>setName(e.target.value)}/>
      <button className="button primary" disabled={!name.trim()}>{ui("保存")}</button>
      <button type="button" className="button quiet" onClick={()=>setSaving(false)}>{ui("取消")}</button>
    </form>)}
  </div>
}
