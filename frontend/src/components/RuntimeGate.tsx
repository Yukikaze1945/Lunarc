// localized-render-sinks
import {ui} from '../i18n/locale'
import {useEffect,useState,type ReactNode} from 'react'
import {Guide} from './Guide'
import {useComponents} from './ComponentPanel'
import {applyTheme,readTheme} from '../services/appearance'

export function RuntimeGate({children}:{children:ReactNode}) {
  const {state,loaded,error}=useComponents()
  const [entered,setEntered]=useState(false)
  const [starting,setStarting]=useState(false)
  const [failure,setFailure]=useState('')
  const [tutorial,setTutorial]=useState(()=>localStorage.getItem('lunarc.guide.v1')!=='done')
  useEffect(()=>{applyTheme(readTheme())},[])
  useEffect(()=>{
    if(!loaded||!state||state.requiresSetup||tutorial||entered||starting)return
    void start()
  },[loaded,state?.requiresSetup,tutorial,entered])
  async function start() {
    if(state?.requiresSetup){setFailure('请先安装基础运行环境和视频处理组件。');return}
    setStarting(true);setFailure('')
    try {
      const result=await (window as any).electronAPI.startComponentBackend()
      if(result?.state!=='ready')throw new Error(result?.message||'后端未启动，请重试')
      localStorage.setItem('lunarc.guide.v1','done');setTutorial(false);setEntered(true)
    }catch(e){setFailure((e as Error).message)}finally{setStarting(false)}
  }
  if(!loaded)return <div className="setup-shell"><header className="project-header"><span className="wordmark">Lunarc</span></header><p className="setup-feedback">{ui("正在检查组件…")}</p></div>
  if(!error&&!state || entered)return children
  return <div className="setup-shell"><header className="project-header"><span className="wordmark">Lunarc</span><span className="setup-caption">{ui("首次使用与组件安装")}</span></header>
    {ui(tutorial||state?.requiresSetup?<Guide setup onFinish={()=>void start()}/>:<div className="workspace-page"><h1>{ui("正在启动工作台")}</h1><button className="button" disabled={starting} onClick={()=>void start()}>{ui("重新启动")}</button></div>)}
    {ui((starting||error||failure)&&<p className="setup-feedback" role={failure||error?'alert':'status'}>{ui(starting?'正在启动工作台…':error||failure)}</p>)}
  </div>
}
