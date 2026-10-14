import {Activity} from 'lucide-react'
import {useSyncExternalStore} from 'react'
import type {SystemInfo} from '../types'
import {subscribePreviewMetrics,previewMetricsSnapshot} from '../services/previewMetrics'
export type TabType = 'home'|'dlss'|'deliver'|'compare'|'diagnostics'|'presets'|'settings'|'about'|'guide'
export function BottomBar({statusText='就绪', queueSummary='', onQueueClick,system}: {statusText?:string; queueSummary?:string; onQueueClick?:()=>void;system?:SystemInfo|null}) {
  const data=useSyncExternalStore(subscribePreviewMetrics,previewMetricsSnapshot)
  const effects=[...(data?.native_input_effect?.stages||[]),...(data?.native_output_effect?.stages||[])]
  const names:Record<string,string>={nr:'NR',dlaa:'DLAA',vsr:'VSR',truehdr:'TrueHDR',vfg:'VFG'}
  const ms=(value:unknown)=>typeof value==='number'&&Number.isFinite(value)?`${value.toFixed(1)} ms`:'—'
  return <footer className="status-bar"><span className="status-message" role="status" title={statusText}><Activity size={13}/>{statusText}</span>
    <div className="status-metrics" aria-label="运行指标">
      {system?.gpu_memory?.process&&<span title={`本进程 DXGI 显存用量 / 系统分配预算；${system.gpu_memory.active?'最近':'上次'}帧间采样。分项见运行与诊断。`}>进程 {(system.gpu_memory.process.local_usage/1024**3).toFixed(2)} / {(system.gpu_memory.process.local_budget/1024**3).toFixed(2)} GiB</span>}
      <span title="整张显卡显存占用，包含其他程序；约每 5 秒采样。">显存 {system?.adapter_memory_used_gb==null?'—':`${system.adapter_memory_used_gb.toFixed(1)} / ${system.adapter_memory_total_gb?.toFixed(1)} GB`}</span>
      <span title={`最近一帧解码及预览处理耗时 ${ms(data?.processing_ms)}，换算为处理帧率；不等同于持续播放帧率。`}>渲染 {data?.cached_display?'缓存':typeof data?.processing_fps==='number'?`${data.processing_fps.toFixed(1)} FPS`:'—'}</span>
      {data?.playing&&<span title="实际呈现的帧率，包含解码、处理和显示等待；不是素材标称帧率。">显示 {Number(data.presentation_fps||0).toFixed(1)} FPS</span>}
      {(data?.render_pipeline||[]).map((stage:string)=>{
        const measured=effects.find((effect:any)=>effect.stage===stage)
        const nr=stage==='nr'&&!measured
        return <span key={stage} title={nr?'DLSSNR 各层模型推理合计；光流、融合和显示耗时另计。':'最近一次完成的原生调用墙钟耗时；不是 GPU 时间戳。'}>{names[stage]||stage}{nr?' 推理':''} {stage==='vfg'?'仅导出':stage==='nr'&&data?.inference_reused?'缓存':ms(measured?.process_ms??(nr?(data?.stages?.nr_total_ms??data?.stages?.dlss_inference):undefined))}</span>
      })}
      {data?.solver&&<span title="数据一致性融合求解耗时">融合 {ms(data.stages?.hdr_fusion??data.stages?.fusion??data.stages?.fusion_solve??data.solver?.solve_ms)}</span>}
    </div>
    <button onClick={onQueueClick} className="status-queue">{queueSummary || '队列空闲'}</button></footer>
}
