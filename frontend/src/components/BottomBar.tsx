import {Activity} from 'lucide-react'
export type TabType = 'home'|'dlss'|'deliver'|'compare'|'diagnostics'|'presets'|'settings'|'about'|'guide'
export function BottomBar({statusText='就绪', queueSummary='', onQueueClick}: {statusText?:string; queueSummary?:string; onQueueClick?:()=>void}) {
  return <footer className="status-bar"><span role="status" title={statusText}><Activity size={13}/>{statusText}</span><button onClick={onQueueClick} className="status-queue">{queueSummary || '队列空闲'}</button></footer>
}
