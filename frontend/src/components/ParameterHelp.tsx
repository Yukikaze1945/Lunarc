import { CircleHelp } from 'lucide-react'
import {useEffect, useId, useRef, useState} from 'react'
import {createPortal} from 'react-dom'
import { PARAMETER_HELP } from '../services/parameterHelp'
import './parameterHelp.css'

export function ParameterHelp({ name, label }: { name: string; label: string }) {
  const help = PARAMETER_HELP[name]
  const [position, setPosition] = useState<{left:number;top:number}|null>(null)
  const button = useRef<HTMLButtonElement>(null)
  const id = useId()
  const show = () => {
    const rect = button.current?.getBoundingClientRect()
    if (rect) setPosition({left:Math.max(8,Math.min(rect.right-250,innerWidth-258)),
      top:Math.max(8,Math.min(rect.bottom+8,innerHeight-125))})
  }
  useEffect(() => {
    const close = () => setPosition(null)
    window.addEventListener('resize', close)
    document.addEventListener('scroll', close, true)
    return () => {window.removeEventListener('resize',close);document.removeEventListener('scroll',close,true)}
  }, [])
  if (!help) return null
  return <span className="parameter-help" onClick={event => event.stopPropagation()}
    onMouseEnter={show} onMouseLeave={()=>setPosition(null)}>
    <button ref={button} type="button" aria-label={`${label}说明`} aria-describedby={position?id:undefined}
      onFocus={show} onBlur={()=>setPosition(null)} onClick={show} onKeyDown={event=>{if(event.key==='Escape')setPosition(null)}}>
      <CircleHelp size={13} aria-hidden="true" />
    </button>
    {position && createPortal(<span id={id} className="parameter-tooltip" role="tooltip" style={position}>{help}</span>,document.body)}
  </span>
}
