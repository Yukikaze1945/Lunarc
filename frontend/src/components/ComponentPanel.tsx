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
  return <section className="panel component-panel"><h2>{basicOnly?'安装基础组件':'组件管理'}</h2>
    <p>{basicOnly?'首次使用需要联网安装。组件保存在本机，软件更新时继续使用。':'根据需要安装。原生 NVIDIA / AMD 光流不需要下载 PyTorch。'}</p>
    {!state?<p>源码运行请按构建文档准备环境；安装版提供在线组件管理。</p>:<>
      <div className="component-list">{state.components.filter(c=>!basicOnly||['base','ffmpeg'].includes(c.id)).map(c=><div className="component-row" key={c.id}>
        <div><h3>{c.label}</h3><p>{c.detail}</p></div><button className={`button ${c.installed?'quiet':'primary'}`} disabled={c.installed||!!state.busy} onClick={()=>install(c.id)}>{c.installed?'已安装':state.busy===c.id?'安装中…':'安装'}</button>
      </div>)}</div>
      <div role="status" aria-live="polite">{state.phase}</div>
      {!!state.busy&&<div className="component-progress">{state.total>0?<><progress aria-label="组件下载进度" max={state.total} value={state.received}/><span>{Math.round(state.received/1048576)} / {Math.round(state.total/1048576)} MB</span></>:<span>下载与安装进行中，请保持网络连接。</span>}<button className="button quiet" onClick={()=>api.cancelComponent()}>暂停</button></div>}
      {state.components.find(c=>c.id==='raft')?.installed&&!basicOnly&&<p>AI 组件安装完成后需要重启软件。</p>}
    </>}
    {(error||actionError||state?.error)&&<p className="feedback" role="alert">{error||actionError||state?.error}</p>}
  </section>
}
