import {useEffect, useRef, useState, type DragEvent} from 'react'
import {Home, Layers3, Film, Activity, Bookmark, Settings2, Info, Aperture, ArrowLeftRight} from 'lucide-react'
import {Header} from './components/Header'
import {BottomBar, type TabType} from './components/BottomBar'
import {DLSSPage} from './pages/DLSSPage'
import {DeliverPage} from './pages/DeliverPage'
import {WorkspacePages} from './pages/WorkspacePages'
import {ComparePage} from './pages/ComparePage'
import type {VideoInfo, DLSSSettings, SystemInfo, QueueState} from './types'
import {fetchSystemInfo, fetchCurrentVideo, loadVideoFile, fetchSettings, saveSettings, fetchQueue} from './services/api'
import {DEFAULT_SETTINGS, completeSettings, readWorkspace, storeWorkspace} from './services/workspace'
import {applyTheme, readTheme} from './services/appearance'
import './App.css'
import {droppedMediaPath} from './services/mediaDrop'
const NAV: {id:TabType;label:string;icon:typeof Home}[] = [
  {id:'home',label:'首页',icon:Home},
  {id:'dlss',label:'重建工作台',icon:Layers3},
  {id:'deliver',label:'导出与队列',icon:Film},
  {id:'compare',label:'视频对比',icon:ArrowLeftRight},
  {id:'diagnostics',label:'运行与诊断',icon:Activity},
  {id:'presets',label:'预设管理',icon:Bookmark},
  {id:'settings',label:'软件设置',icon:Settings2},
  {id:'about',label:'关于 Lunarc',icon:Info},
]
export function App() {
  const [dragging, setDragging] = useState(false)
  const dragDepth = useRef(0)
  const importRevision = useRef(0)
  const importQueue = useRef<Promise<void>>(Promise.resolve())
  const [activeTab,setActiveTab]=useState<TabType>('dlss')
  const [comparePair,setComparePair]=useState<{original:string;exported:string}|null>(null)
  useEffect(()=>{const open=(event:Event)=>{const pair=(event as CustomEvent).detail;
    if(pair?.original&&pair?.exported){setComparePair(pair);setActiveTab('compare')}};
    window.addEventListener('lunarc:compare',open);return()=>window.removeEventListener('lunarc:compare',open)},[])
  const [theme,setTheme]=useState(readTheme)
  useEffect(()=>applyTheme(theme),[theme])
  const [video,setVideo]=useState<VideoInfo|null>(null)
  const [system,setSystem]=useState<SystemInfo|null>(null)
  const [settings,setSettings]=useState<DLSSSettings>(DEFAULT_SETTINGS)
  const [recent,setRecent]=useState<string[]>(()=>readWorkspace()?.recent||[])
  const [currentFrame,setCurrentFrame]=useState(0)
  const [statusText,setStatusText]=useState('连接本地工作台…')
  const [ready,setReady]=useState(false)
  const [queue,setQueue]=useState<QueueState|null>(null)
  useEffect(()=>{
    let cancelled=false, initialized=false, inFlight=false
    const init=async()=>{
      if(inFlight||cancelled)return
      inFlight=true
      try {
        const [sys,v,s]=await Promise.all([fetchSystemInfo(),fetchCurrentVideo(),fetchSettings()])
        if(cancelled)return
        setSystem(sys);if(v)setVideo(v)
        const draft=readWorkspace()
        if(draft){try{const restored=await saveSettings(draft.settings);if(!cancelled)setSettings(completeSettings(restored));setStatusText('已恢复参数')}catch(e:any){setSettings(completeSettings(s));setStatusText(`草稿未应用：${e.message}；保留后端参数`)}}
        else{
          const current=s.style===3?await saveSettings({style:DEFAULT_SETTINGS.style}):s
          if(cancelled)return
          setSettings(completeSettings(current));setStatusText('工作台就绪')
        }
        if(!cancelled){initialized=true;setReady(true)}
      }catch(e:any){
        const startup=await window.electronAPI?.getBackendStartup?.().catch(()=>null)
        if(!cancelled)setStatusText(startup?.state==='error'?startup.message:`后端连接失败：${e.message} · 正在重试`)
      }finally{inFlight=false}
    }
    void init()
    const timer=setInterval(()=>{if(!initialized){void init();return}void fetchSystemInfo().then(s=>{if(!cancelled)setSystem(s)}).catch(()=>{})},5000)
    const unsubscribe=window.electronAPI?.onBackendExit?.(info=>{if(!cancelled){setReady(false);setStatusText(`后端已退出 (${info.code})，请重启应用`)}})
    return()=>{cancelled=true;clearInterval(timer);unsubscribe?.()}
  },[])
  useEffect(()=>{let stopped=false;const poll=async()=>{try{const q=await fetchQueue();if(stopped)return;setQueue(q)
    const terminal=q.jobs.filter(j=>['completed','failed','cancelled','skipped'].includes(j.status)).map(j=>({id:j.id,name:j.name,status:j.status,input_path:j.input_path,output_path:j.output_path,error:j.error,recorded:new Date().toISOString()}))
    if(terminal.length){let history:any[]=[];try{const h=JSON.parse(localStorage.getItem('lunarc.history.v1')||'[]');if(Array.isArray(h))history=h}catch{}
      const merged=[...terminal.map(j=>({...j,recorded:history.find(h=>h.id===j.id)?.recorded||j.recorded})),...history.filter(h=>!terminal.some(j=>j.id===h.id))].slice(0,100);const serialized=JSON.stringify(merged);if(serialized!==localStorage.getItem('lunarc.history.v1'))localStorage.setItem('lunarc.history.v1',serialized)}
    }catch{}};void poll();const t=setInterval(poll,1500);return()=>{stopped=true;clearInterval(t)}},[])
  useEffect(()=>{if(ready){try{storeWorkspace(settings,recent)}catch{setStatusText('本机草稿无法保存；当前后端参数仍保留')}}},[settings,recent,ready])
  const importFile=async(path:string)=>{
    const revision=++importRevision.current;setStatusText('正在读取素材…')
    // Serialize source replacement on the backend too, and skip obsolete drops.
    const request=importQueue.current.then(async()=>{
      if(revision!==importRevision.current)return
      try{const loaded=await loadVideoFile(path);if(revision!==importRevision.current)return;setVideo(loaded);setCurrentFrame(0);setRecent(r=>[path,...r.filter(x=>x!==path)].slice(0,8));setActiveTab('dlss');setStatusText(`已载入 ${loaded.filename}`)}catch(e:any){if(revision===importRevision.current)setStatusText(`素材载入失败：${e.message}`)}
    })
    importQueue.current=request.catch(()=>{});await request
  }
  const hasFiles = (event: DragEvent) => Array.from(event.dataTransfer.types).includes('Files')
  const dropMedia = (event: DragEvent) => {
    if (!hasFiles(event)) return
    event.preventDefault(); event.stopPropagation(); dragDepth.current=0; setDragging(false)
    const resolve = window.electronAPI?.getPathForFile
    if (!resolve) { setStatusText('请在桌面程序中拖入素材'); return }
    try {
      const path=droppedMediaPath(event.dataTransfer.files, resolve)
      if (path) void importFile(path)
      else setStatusText('请拖入视频或图片文件')
    } catch (error: any) { setStatusText(`素材载入失败：${error.message}`) }
  }
  const applySettings=async(next:DLSSSettings)=>{const saved=await saveSettings(completeSettings(next));setSettings(completeSettings(saved));setStatusText('参数已验证并保存')}
  const running=queue?.jobs.find(j=>j.status==='running')
  const pending=queue?.jobs.filter(j=>j.status==='pending').length||0
  const queueSummary=running?`${running.name} · ${Math.round(running.progress)}%`:pending?`${pending} 个任务等待导出`:'队列空闲'
  return <div className="app-shell" onDragEnter={event=>{if(hasFiles(event)){event.preventDefault();if((event.target as Element).closest?.('.dlssnr-runtime')){dragDepth.current=0;setDragging(false);return}dragDepth.current++;setDragging(true)}}}
    onDragOver={event=>{if(hasFiles(event)){event.preventDefault();event.dataTransfer.dropEffect='copy'}}}
    onDragLeave={event=>{if(hasFiles(event)){event.preventDefault();dragDepth.current=Math.max(0,dragDepth.current-1);if(!dragDepth.current)setDragging(false)}}}
    onDrop={dropMedia}>
    {dragging&&<div className="media-drop-overlay" role="status"><span>松开以载入素材</span></div>}
    <aside className="navigation"><div className="rail-mark" aria-hidden="true"><Aperture size={23}/></div><nav aria-label="主导航">{NAV.map(n=><button key={n.id} aria-label={n.label} title={n.label} className={activeTab===n.id?'active':''} onClick={()=>setActiveTab(n.id)} aria-current={activeTab===n.id?'page':undefined}><n.icon size={20}/></button>)}</nav></aside>
    <div className="app-content"><Header activeTab={activeTab} onNavigate={setActiveTab} theme={theme} onThemeChange={()=>setTheme(t=>t==='dark'?'light':'dark')} onFileLoaded={importFile} onError={setStatusText}/><main className="page-host">
      <div className={`mounted-page ${activeTab==='dlss'?'':'hidden'}`}><DLSSPage video={video} settings={settings} onSettingsChange={s=>setSettings(completeSettings(s))} currentFrame={currentFrame} setCurrentFrame={setCurrentFrame} onStatusChange={setStatusText} isActive={activeTab==='dlss'}/></div>
      <div className={`mounted-page ${activeTab==='deliver'?'':'hidden'}`}><DeliverPage video={video} settings={settings} onSettingsChange={s=>setSettings(completeSettings(s))} onStatusChange={setStatusText}/></div>
      {activeTab==='compare'&&<ComparePage initialPair={comparePair}/>}
      {!['dlss','deliver','compare'].includes(activeTab)&&<WorkspacePages page={activeTab} video={video} system={system} settings={settings} recent={recent} onImport={importFile} onNavigate={setActiveTab} onApply={applySettings} onStatus={setStatusText}/>}
    </main><BottomBar statusText={statusText} queueSummary={queueSummary} onQueueClick={()=>setActiveTab('deliver')}/></div>
  </div>
}
export default App
