import {useEffect,useRef,type RefObject} from 'react'

export function wheelZoom(initial:number,delta:number):number {
  return Math.round(Math.max(.25,Math.min(8,initial*2**(-delta/600)))*1000)/1000
}

// React wheel handlers can be passive. Cancel only gestures consumed by the image.
export function useImageWheel(element:RefObject<HTMLElement|null>,enabled:boolean,
    onZoom:(delta:number)=>void,onScroll:(delta:number)=>boolean) {
  const current=useRef({onZoom,onScroll});current.current={onZoom,onScroll}
  useEffect(()=>{
    const canvas=element.current
    if(!enabled||!canvas)return
    const wheel=(event:WheelEvent)=>{
      if(event.ctrlKey)return
      const delta=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?canvas.clientHeight:1)
      if(!delta)return
      if(event.altKey){event.preventDefault();current.current.onZoom(delta)}
      else if(current.current.onScroll(delta))event.preventDefault()
    }
    canvas.addEventListener('wheel',wheel,{passive:false})
    return()=>canvas.removeEventListener('wheel',wheel)
  },[element,enabled])
}
