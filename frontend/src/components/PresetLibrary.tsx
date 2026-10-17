// localized-render-sinks
import {ui} from '../i18n/locale'
import {useEffect,useSyncExternalStore,useState} from 'react'
import {Trash2,Upload,Download} from 'lucide-react'
import {presetSnapshot,subscribePresets,deletePreset,importPresets,presetChanges} from '../services/presets'
import {completeSettings} from '../services/workspace'
import './presets.css'

export function PresetLibrary() {
  const presets=useSyncExternalStore(subscribePresets,presetSnapshot),[message,setMessage]=useState('')
  useEffect(()=>{const report=(event:Event)=>setMessage((event as CustomEvent).detail.message);window.addEventListener('lunarc:preset-error',report);return()=>window.removeEventListener('lunarc:preset-error',report)},[])
  const importFile=async(file:File)=>{try{const p=JSON.parse(await file.text());if(p.version!==1||!p.settings||typeof p.settings!=='object'||Array.isArray(p.settings))throw new Error('需要 Lunarc 参数文件');
    await importPresets([{name:p.name||file.name.replace(/\.json$/i,''),settings:completeSettings(p.settings)}]);setMessage('预设已导入')}catch(e:any){setMessage(e.message)}}
  return <div className="workspace-page preset-library"><div className="preset-library-head"><h1>{ui("预设管理")}</h1>
    <label className="button"><Upload size={14}/>{ui("导入预设")}<input type="file" aria-label={ui("导入预设文件")} accept=".json" hidden onChange={e=>{if(e.target.files?.[0])void importFile(e.target.files[0]);e.target.value=''}}/></label></div>
    {ui(presets.length?presets.map(p=>{const changes=presetChanges(p.settings);return <details className="preset-entry" key={p.name}>
      <summary><strong>{p.name}</strong><span>{ui(changes.length)}{ui(" 项改动")}</span>
        <button type="button" className="button quiet" aria-label={ui(`导出预设 ${p.name}`)} onClick={e=>{e.preventDefault();const url=URL.createObjectURL(new Blob([JSON.stringify({version:1,...p},null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`${p.name}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}}><Download size={14}/></button>
        <button type="button" className="button quiet" aria-label={ui(`删除预设 ${p.name}`)} onClick={async e=>{e.preventDefault();try{await deletePreset(p.name)}catch(error:any){setMessage(error.message)}}}><Trash2 size={14}/></button><span className="fold-sign"/>
      </summary>
      {ui(changes.length?<table><thead><tr><th>{ui("参数")}</th><th>{ui("默认")}</th><th>{ui("预设")}</th><th>{ui("改动幅度")}</th></tr></thead><tbody>{ui(changes.map(c=><tr key={c.key}><td>{ui(c.label)}</td><td>{ui(c.before)}</td><td>{ui(c.after)}</td><td>{ui(c.delta!==undefined?`${c.delta>0?'+':''}${Number(c.delta.toFixed(4))}${c.percent!==undefined?`（${c.percent>0?'+':''}${Number(c.percent.toFixed(1))}%）`:''}`:'—')}</td></tr>))}</tbody></table>:<p>{ui("与默认参数一致")}</p>)}
    </details>}):<p className="muted">{ui("还没有预设。在重建工作台保存当前参数后，会显示在这里。")}</p>)}
    <p role="status" className="feedback">{ui(message)}</p>
  </div>
}
