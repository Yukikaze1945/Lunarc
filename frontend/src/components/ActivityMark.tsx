import {Aperture} from 'lucide-react'
import {useEffect,useLayoutEffect,useRef,useState} from 'react'
import {ui} from '../i18n/locale'
import type {ActivityState} from '../services/activityState'
export function ActivityMark({state}:{state:ActivityState}){
  const glyph=useRef<SVGSVGElement>(null),motion=useRef<Animation|null>(null)
  const [reduced,setReduced]=useState(()=>matchMedia('(prefers-reduced-motion:reduce)').matches||document.documentElement.dataset.reducedMotion==='true')
  useEffect(()=>{
    const query=matchMedia('(prefers-reduced-motion:reduce)')
    const update=()=>setReduced(query.matches||document.documentElement.dataset.reducedMotion==='true')
    query.addEventListener('change',update)
    const observer=new MutationObserver(update)
    observer.observe(document.documentElement,{attributes:true,attributeFilter:['data-reduced-motion']})
    return()=>{query.removeEventListener('change',update);observer.disconnect()}
  },[])
  useLayoutEffect(()=>{
    const node=glyph.current
    if(!node)return
    const angle=()=>{
      const matrix=new DOMMatrixReadOnly(getComputedStyle(node).transform)
      return (Math.atan2(matrix.b,matrix.a)*180/Math.PI+360)%360
    }
    const start=angle(),previous=motion.current
    let speed=0
    // Sample the current animation without painting an intermediate frame.
    // Matching the tail's initial velocity avoids a jerk when work completes.
    if(previous&&typeof previous.currentTime==='number'){
      const time=previous.currentTime
      previous.currentTime=time+4
      speed=((angle()-start+360)%360)/.004
      previous.currentTime=time
    }
    node.style.transform=`rotate(${start}deg)`
    previous?.cancel();motion.current=null
    if(reduced){node.style.transform='rotate(0deg)';node.dataset.motion='rest';return}
    const spinning=['loading','recovering','exporting'].includes(state)
    if(spinning){
      node.dataset.motion='spinning'
      const frames=state==='loading'?
        [{transform:`rotate(${start}deg)`,offset:0,easing:'cubic-bezier(.2,.05,.35,1)'},
          {transform:`rotate(${start+720}deg)`,offset:.4,easing:'linear'},
          {transform:`rotate(${start+800}deg)`,offset:.65,easing:'cubic-bezier(.6,0,.8,1)'},
          {transform:`rotate(${start+1440}deg)`,offset:1}]:
        [{transform:`rotate(${start}deg)`},{transform:`rotate(${start+360}deg)`}]
      motion.current=node.animate(frames,{duration:state==='loading'?5000:state==='recovering'?1800:12000,iterations:Infinity})
      return
    }
    if(!previous||(start<.05&&speed<1)){node.style.transform='rotate(0deg)';node.dataset.motion='rest';return}
    const end=Math.ceil((start+Math.max(4,speed*.18))/360)*360,distance=end-start
    const duration=Math.max(550,Math.min(1600,2200*distance/Math.max(speed,60)))
    const slope=speed*duration/1000/distance,x=Math.min(.25,.8/Math.max(slope,.01))
    node.dataset.motion='settling'
    const tail=node.animate([{transform:`rotate(${start}deg)`},{transform:`rotate(${end}deg)`}],
      {duration,easing:`cubic-bezier(${x},${slope*x},.6,1)`,fill:'forwards'})
    motion.current=tail
    void tail.finished.then(()=>{
      if(motion.current!==tail)return
      node.style.transform='rotate(0deg)';node.dataset.motion='rest';motion.current=null;tail.cancel()
    }).catch(()=>{})
  },[state,reduced])
  useLayoutEffect(()=>()=>{motion.current?.cancel()},[])
  const label={idle:'就绪',loading:'处理中',error:'处理失败',recovering:'正在重启后端',exporting:'正在导出'}[state]
  return <div className="rail-mark" data-state={state} role="status" aria-label={ui(label)} title={ui(label)}><span className="activity-glyph"><Aperture ref={glyph} size={23}/></span></div>
}
