import {useEffect,useRef,useState,type PointerEvent} from 'react'
import {useImageWheel,wheelZoom} from '../services/useImageWheel'

type Point={x:number;y:number}
type View='orig'|'dlss'|'split'|'dual'|'flow'
export function ImageViewport({src,paired=false,view='dlss',zoom,onZoom,pan,onPan,
    split=.5,onSplit}:{src:string;paired?:boolean;view?:View;zoom:number;onZoom:(value:number)=>void;
    pan:Point;onPan:(value:Point)=>void;split?:number;onSplit?:(value:number)=>void}) {
  const box=useRef<HTMLDivElement>(null),canvas=useRef<HTMLCanvasElement>(null)
  const [image,setImage]=useState<HTMLImageElement|null>(null)
  const [bounds,setBounds]=useState({width:1,height:1})
  const drag=useRef<{id:number;x:number;y:number;wipe:boolean}|null>(null)
  const current=useRef({zoom,pan});current.current={zoom,pan}
  useEffect(()=>{
    let alive=true;setImage(null)
    const loaded=new Image();loaded.onload=()=>{if(alive)setImage(loaded)};loaded.src=src
    return()=>{alive=false}
  },[src])
  useEffect(()=>{
    if(!box.current)return
    const observer=new ResizeObserver(([entry])=>setBounds({width:entry.contentRect.width,height:entry.contentRect.height}))
    observer.observe(box.current);return()=>observer.disconnect()
  },[])
  const width=image?image.naturalWidth/(paired?2:1):1,height=image?.naturalHeight||1
  const columns=paired&&view==='dual'?2:1
  const fit=Math.min(bounds.width/(width*columns),bounds.height/height)
  const renderedWidth=width*fit*zoom,renderedHeight=height*fit*zoom
  const shifted=(value:number,size:number,viewport:number)=>
    Math.max(-Math.max(0,size-viewport)/2,Math.min(Math.max(0,size-viewport)/2,(.5-value)*size))
  const changePan=(dx:number,dy:number)=>{
    const previous=current.current.pan
    const clamp=(value:number,size:number,viewport:number)=>{
      const limit=Math.max(0,1-viewport/size)/2
      return Math.max(.5-limit,Math.min(.5+limit,value))
    }
    const next={x:clamp(previous.x-dx/(renderedWidth*columns),renderedWidth*columns,bounds.width),
      y:clamp(previous.y-dy/renderedHeight,renderedHeight,bounds.height)}
    current.current.pan=next;onPan(next)
  }
  useImageWheel(box,true,delta=>{
    const value=wheelZoom(current.current.zoom,delta);current.current.zoom=value;onZoom(value)
  },delta=>{
    if(renderedHeight<=bounds.height)return false
    changePan(0,-delta);return true
  })
  useEffect(()=>{
    const target=canvas.current,context=target?.getContext('2d')
    if(!target||!context)return
    const dpi=devicePixelRatio||1
    target.width=Math.round(bounds.width*dpi);target.height=Math.round(bounds.height*dpi)
    context.scale(dpi,dpi);context.clearRect(0,0,bounds.width,bounds.height)
    if(!image)return
    const x=(bounds.width-renderedWidth*columns)/2+shifted(pan.x,renderedWidth*columns,bounds.width)
    const y=(bounds.height-renderedHeight)/2+shifted(pan.y,renderedHeight,bounds.height)
    const draw=(index:number,left=x)=>context.drawImage(image,index*width,0,width,height,left,y,renderedWidth,renderedHeight)
    if(!paired)draw(0)
    else if(view==='dual'){draw(0);draw(1,x+renderedWidth)}
    else if(view==='orig')draw(0)
    else if(view==='split'){
      draw(0);context.save();context.beginPath();context.rect(bounds.width*split,0,bounds.width*(1-split),bounds.height)
      context.clip();draw(1);context.restore()
    }else draw(1)
  },[image,bounds,zoom,pan,view,split,paired,renderedWidth,renderedHeight,width,height,columns])
  const wipe=(event:PointerEvent<HTMLDivElement>)=>{
    const rect=box.current!.getBoundingClientRect();onSplit?.(Math.max(.01,Math.min(.99,(event.clientX-rect.left)/rect.width)))
  }
  const finish=(event:PointerEvent<HTMLDivElement>)=>{
    if(drag.current?.id!==event.pointerId)return
    drag.current=null
    if(event.currentTarget.hasPointerCapture(event.pointerId))event.currentTarget.releasePointerCapture(event.pointerId)
  }
  return <div ref={box} className="image-viewport" aria-label="图片预览画布"
    title="Alt＋滚轮缩放；左键拖动画面；滚轮上下移动；双击恢复适配"
    onDoubleClick={()=>{onZoom(1);onPan({x:.5,y:.5})}}
    onPointerDown={event=>{
      if(event.button!==0)return
      event.preventDefault();event.stopPropagation()
      const rect=box.current!.getBoundingClientRect()
      const isWipe=paired&&view==='split'&&Math.abs(event.clientX-rect.left-split*rect.width)<18
      drag.current={id:event.pointerId,x:event.clientX,y:event.clientY,wipe:isWipe}
      event.currentTarget.setPointerCapture(event.pointerId);if(isWipe)wipe(event)
    }} onPointerMove={event=>{
      const previous=drag.current;if(!previous||previous.id!==event.pointerId)return
      event.stopPropagation()
      if(previous.wipe)wipe(event)
      else changePan(event.clientX-previous.x,event.clientY-previous.y)
      previous.x=event.clientX;previous.y=event.clientY
    }} onPointerUp={finish} onPointerCancel={finish} onLostPointerCapture={()=>{drag.current=null}}>
    <canvas ref={canvas}/>
    {paired&&view==='split'&&<div className="split-marker" aria-hidden="true" style={{left:`${split*100}%`}}/>}
  </div>
}
