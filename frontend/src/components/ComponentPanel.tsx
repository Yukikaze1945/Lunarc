// localized-render-sinks
import {ui} from '../i18n/locale'
import {useEffect,useState} from 'react'
import type {ComponentState} from '../types/components'
import './guide.css'

export function useComponents() {
  const api=(window as any).electronAPI
  const [state,setState]=useState<ComponentState|null>(null)
  const [loaded,setLoaded]=useState(false)
  const [error,setError]=useState('')
  useEffect(()=>{
    if(!api?.getComponents){setLoaded(true);return}
    let active=true,revision=0
    const stop=api.onComponents((s:ComponentState)=>{revision++;if(active)setState(s)})
    const initial=revision
    api.getComponents().then((s:ComponentState|null)=>{if(active){if(initial===revision)setState(s);setLoaded(true)}}).catch((e:Error)=>{if(active){setError(e.message);setLoaded(true)}})
    return()=>{active=false;stop()}
  },[api])
  return {state,loaded,error}
}
export function ComponentPanel({basicOnly=false}:{basicOnly?:boolean}) {
  const {state,error}=useComponents()
  const [actionError,setActionError]=useState('')
  const api=(window as any).electronAPI
  const install=async(id:string)=>{setActionError('');try{await api.installComponent(id)}catch(e){setActionError((e as Error).message)}}
  return <section className="panel component-panel"><h2>{ui(basicOnly?'安装基础组件':'组件管理')}</h2>
    <p>{ui(basicOnly?'首次使用需要联网安装。组件保存在本机，软件更新时继续使用。':'根据需要安装。原生 NVIDIA / AMD 光流不需要下载 PyTorch。')}</p>
    {ui(!state?<p>{ui("源码运行请按构建文档准备环境；安装版提供在线组件管理。")}</p>:<>
      <div className="component-list">{ui(state.components.filter(c=>!basicOnly||['base','ffmpeg'].includes(c.id)).map(c=><div className="component-row" key={c.id}>
        <div><h3>{ui(c.label)}</h3><p>{ui(c.detail)}</p></div><button className={`button ${c.installed?'quiet':'primary'}`} disabled={c.installed||!!state.busy} onClick={()=>install(c.id)}>{ui(c.installed?'已安装':state.busy===c.id?'安装中…':'安装')}</button>
      </div>))}</div>
      <div role="status" aria-live="polite">{ui(state.phase)}</div>
      {ui(!!state.busy&&<div className="component-progress">{ui(state.total>0?<><progress aria-label={ui("组件下载进度")} max={state.total} value={state.received}/><span>{ui(Math.round(state.received/1048576))} / {ui(Math.round(state.total/1048576))} MB</span></>:<span>{ui("下载与安装进行中，请保持网络连接。")}</span>)}<button className="button quiet" onClick={()=>api.cancelComponent()}>{ui("暂停")}</button></div>)}
      {ui(state.components.find(c=>c.id==='raft')?.installed&&!basicOnly&&<p>{ui("AI 组件安装完成后需要重启软件。")}</p>)}
    </>)}
    {ui((error||actionError||state?.error)&&<p className="feedback" role="alert">{ui(error||actionError||state?.error)}</p>)}
  </section>
}
