// localized-render-sinks
import {ui} from '../i18n/locale'
import {useEffect, useState} from 'react'
import {Download, FolderOpen, RefreshCw} from 'lucide-react'
import type {UpdateState} from '../types/appUpdates'

export function AppUpdatePanel() {
  const api=(window as any).electronAPI
  const [state,setState]=useState<UpdateState|null>(null)
  const [preview,setPreview]=useState(()=>localStorage.getItem('lunarc.updates.preview')!=='false')
  const [error,setError]=useState('')
  useEffect(()=>{
    if (!api?.getAppUpdate) return
    let active=true, revision=0
    const stop=api.onAppUpdate((s:UpdateState)=>{revision++; if(active)setState(s)})
    const initial=revision
    api.getAppUpdate().then((s:UpdateState)=>{if(active&&revision===initial)setState(s)}).catch((e:Error)=>{if(active)setError(e.message)})
    return()=>{active=false;stop()}
  },[api])
  async function action(fn:()=>Promise<unknown>) {setError('');try {await fn()} catch(e) {setError((e as Error).message)}}
  const busy=state?.phase==='checking'||state?.phase==='downloading'
  const size=(bytes:number)=>bytes<1024**3?`${Math.round(bytes/1024**2)} MB`:`${(bytes/1024**3).toFixed(2)} GB`
  return <section className="panel app-update-panel">
    <div className="inline-form"><h2>{ui("软件更新")}</h2><span className="badge">{ui(state?`${state.currentVersion}${state.packaged?'':' · 源码运行'}`:'浏览器运行')}</span></div>
    {ui(!api?.getAppUpdate?<p>{ui("请在桌面版中检查和下载更新。")}</p>:<>
      <label className="check-field"><input type="checkbox" checked={preview} disabled={busy} onChange={e=>{setPreview(e.target.checked);localStorage.setItem('lunarc.updates.preview',String(e.target.checked))}}/>{ui("包含预发布版本")}</label>
      <p role="status" aria-live="polite">{ui(state?.phase==='checking'?'正在检查…':state?.phase==='current'?'当前没有更新的安装包':state?.release?`可用版本 ${state.release.version}${state.release.preview?' · 预发布':''}`:'从 Lunarc 官方 GitHub 发布页获取更新')}</p>
      {ui(state?.phase==='downloading'&&<div><progress aria-label={ui("更新下载进度")} value={state.received} max={state.total}/><p>{ui(size(state.received))} / {ui(size(state.total))}</p></div>)}
      <div className="inline-form">
        <button className="button" disabled={busy} onClick={()=>action(()=>api.checkAppUpdate(preview))}><RefreshCw size={15}/>{ui("检查更新")}</button>
        {ui(state?.release&&state.phase!=='ready'&&<button className="button primary" disabled={busy} onClick={()=>action(()=>api.downloadAppUpdate())}><Download size={15}/>{ui(state.phase==='paused'||state.phase==='error'?'继续下载':'下载更新')} · {ui(size(state.total))}</button>)}
        {ui(state?.phase==='downloading'&&<button className="button" onClick={()=>action(()=>api.cancelAppUpdate())}>{ui("暂停")}</button>)}
        {ui(state?.phase==='ready'&&<button className="button primary" onClick={()=>action(()=>api.openAppUpdateDirectory())}><FolderOpen size={15}/>{ui("打开安装目录")}</button>)}
        {ui(state?.release&&<a className="button quiet" href={state.release.url} target="_blank" rel="noreferrer">{ui("发行说明 ↗")}</a>)}
      </div>
      {ui(state?.phase==='ready'&&<p>{ui("全部文件已校验。")}{ui(state.release?.assets.some(a=>a.name.endsWith('Installer.zip'))?'解压 Installer.zip，退出工作台后运行 Install-Lunarc.cmd。':'退出工作台后运行 Setup.exe。')}{ui("已安装的运行组件会保留。")}</p>)}
      {ui((error||state?.error)&&<p className="feedback" role="alert">{ui(error||state?.error)}</p>)}
    </>)}
  </section>
}
