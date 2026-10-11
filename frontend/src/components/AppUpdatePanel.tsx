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
  const size=(bytes:number)=>`${(bytes/1024**3).toFixed(2)} GB`
  return <section className="panel app-update-panel">
    <div className="inline-form"><h2>软件更新</h2><span className="badge">{state?`${state.currentVersion}${state.packaged?'':' · 源码运行'}`:'浏览器运行'}</span></div>
    {!api?.getAppUpdate?<p>请在桌面版中检查和下载更新。</p>:<>
      <label className="check-field"><input type="checkbox" checked={preview} disabled={busy} onChange={e=>{setPreview(e.target.checked);localStorage.setItem('lunarc.updates.preview',String(e.target.checked))}}/>包含预发布版本</label>
      <p role="status" aria-live="polite">{state?.phase==='checking'?'正在检查…':state?.phase==='current'?'未发现更新的完整安装包':state?.release?`可用版本 ${state.release.version}${state.release.preview?' · 预发布':''}`:'从 Lunarc 官方 GitHub 发布页获取更新'}</p>
      {state?.phase==='downloading'&&<div><progress aria-label="更新下载进度" value={state.received} max={state.total}/><p>{size(state.received)} / {size(state.total)}</p></div>}
      <div className="inline-form">
        <button className="button" disabled={busy} onClick={()=>action(()=>api.checkAppUpdate(preview))}><RefreshCw size={15}/>检查更新</button>
        {state?.release&&state.phase!=='ready'&&<button className="button primary" disabled={busy} onClick={()=>action(()=>api.downloadAppUpdate())}><Download size={15}/>{state.phase==='paused'||state.phase==='error'?'继续下载':'下载更新'} · {size(state.total)}</button>}
        {state?.phase==='downloading'&&<button className="button" onClick={()=>action(()=>api.cancelAppUpdate())}>暂停</button>}
        {state?.phase==='ready'&&<button className="button primary" onClick={()=>action(()=>api.openAppUpdateDirectory())}><FolderOpen size={15}/>打开安装目录</button>}
        {state?.release&&<a className="button quiet" href={state.release.url} target="_blank" rel="noreferrer">发行说明 ↗</a>}
      </div>
      {state?.phase==='ready'&&<p>全部文件已校验。{state.release?.assets.some(a=>a.name.endsWith('Installer.zip'))?'解压 Installer.zip，退出工作台后运行 Install-Lunarc.cmd。':'退出工作台后运行 Setup.exe。'}更新需重新下载安装包。</p>}
      {(error||state?.error)&&<p className="feedback" role="alert">{error||state?.error}</p>}
    </>}
  </section>
}
