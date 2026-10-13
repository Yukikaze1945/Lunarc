import {useEffect,useRef,useState} from 'react'
import type {RefObject} from 'react'
import {localToken} from './api'
let layoutQueue:Promise<unknown>=Promise.resolve()

// Only layout/control metadata crosses HTTP. Pictures stay in the GPU owner.
export function useNativePreview(element:RefObject<HTMLDivElement|null>,enabled:boolean,
    view:string,zoom:number,pan:{x:number;y:number}) {
  const [ready,setReady]=useState(false)
  const [error,setError]=useState<string|null>(null)
  const [hdrActive,setHdrActive]=useState<boolean|null>(null)
  const current=useRef({view,zoom,pan});current.current={view,zoom,pan}
  const update=useRef<()=>void>(()=>{})
  useEffect(()=>{
    if(!enabled||view==='flow'||!window.electronAPI?.getPreviewWindow){setReady(false);setHdrActive(null);setError(null);return}
    let disposed=false,raf=0,busy=false,pending=false,parent='',observer:ResizeObserver|undefined
    const send=async(body:unknown)=>{
      const operation=layoutQueue.catch(()=>{}).then(async()=>{const response=await fetch('http://127.0.0.1:18923/api/preview/native',{
        method:'POST',headers:{'Content-Type':'application/json','X-DLSS-Token':localToken()},body:JSON.stringify(body)})
      if(!response.ok){const data=await response.json();throw new Error(data.detail||'原生预览连接失败')}})
      layoutQueue=operation;await operation
    }
    const layout=async()=>{
      if(disposed||!parent||!element.current)return
      if(busy){pending=true;return}busy=true
      try{
        do{pending=false;const bounds=element.current.getBoundingClientRect(),state=current.current,scale=devicePixelRatio
          await send({parent,x:Math.round(bounds.left*scale),y:Math.round(bounds.top*scale),
            width:Math.max(1,Math.round(bounds.width*scale)),height:Math.max(1,Math.round(bounds.height*scale)),
            visible:true,view:state.view,zoom:Math.min(32,Math.max(.1,state.zoom)),pan_x:state.pan.x,pan_y:state.pan.y})
          if(!disposed){setReady(true);setError(null)}
        }while(pending&&!disposed)
      }catch(e:any){if(!disposed){setReady(false);setError(e.message)}}finally{busy=false}
    }
    const schedule=()=>{cancelAnimationFrame(raf);raf=requestAnimationFrame(()=>void layout())}
    update.current=schedule
    let statusBusy=false
    const timer=setInterval(()=>{
      if(disposed||!parent||statusBusy)return;statusBusy=true
      void fetch('http://127.0.0.1:18923/api/preview/native',{headers:{'X-DLSS-Token':localToken()}})
        .then(async response=>{if(!response.ok)return;const data=await response.json();if(!disposed&&data.native)setHdrActive(Boolean(data.hdr_active))})
        .catch(()=>{}).finally(()=>{statusBusy=false})
    },1000)
    void window.electronAPI.getPreviewWindow().then(handle=>{
      if(disposed||!handle)return;parent=handle.parent
      observer=new ResizeObserver(schedule);if(element.current)observer.observe(element.current)
      window.addEventListener('resize',schedule);window.addEventListener('scroll',schedule,true)
      document.addEventListener('fullscreenchange',schedule);schedule()
    }).catch(e=>{if(!disposed)setError(e.message)})
    return()=>{disposed=true;setReady(false);clearInterval(timer);cancelAnimationFrame(raf);observer?.disconnect()
      window.removeEventListener('resize',schedule);window.removeEventListener('scroll',schedule,true)
      document.removeEventListener('fullscreenchange',schedule)
      // Preserve ordering: detach after the last outstanding layout completes.
      void send(null).catch(()=>{})
    }
  },[enabled,view==='flow',element])
  useEffect(()=>{update.current()},[view,zoom,pan])
  return {ready,error,hdrActive}
}
