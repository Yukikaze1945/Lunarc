// localized-render-sinks
import {ui} from '../i18n/locale'
import {Activity} from 'lucide-react'
import {useSyncExternalStore} from 'react'
import type {SystemInfo} from '../types'
import {subscribePreviewMetrics,previewMetricsSnapshot} from '../services/previewMetrics'
export type TabType = 'home'|'dlss'|'deliver'|'image-deliver'|'compare'|'diagnostics'|'presets'|'settings'|'about'|'guide'
export function BottomBar({statusText='就绪', queueSummary='', onQueueClick,system}: {statusText?:string; queueSummary?:string; onQueueClick?:()=>void;system?:SystemInfo|null}) {
  const data=useSyncExternalStore(subscribePreviewMetrics,previewMetricsSnapshot)
  const effects=[...(data?.native_input_effect?.stages||[]),...(data?.native_output_effect?.stages||[])]
  const names:Record<string,string>={nr:'NR',dlaa:'DLAA',vsr:'VSR',truehdr:'TrueHDR',vfg:'VFG'}
  const ms=(value:unknown)=>typeof value==='number'&&Number.isFinite(value)?`${value.toFixed(1)} ms`:'—'
  return <footer className="status-bar"><span className="status-message" role="status" title={ui(statusText)}><Activity size={13}/>{ui(statusText)}</span>
    <div className="status-metrics" aria-label={ui("运行指标")}>
      {ui(system?.gpu_memory?.process&&<span title={ui(`本进程 DXGI 显存用量 / 系统分配预算；${system.gpu_memory.active?'最近':'上次'}帧间采样。分项见运行与诊断。`)}>{ui("进程 ")}{ui((system.gpu_memory.process.local_usage/1024**3).toFixed(2))} / {ui((system.gpu_memory.process.local_budget/1024**3).toFixed(2))} GiB</span>)}
      <span title={ui("整张显卡显存占用，包含其他程序；约每 5 秒采样。")}>{ui("显存 ")}{ui(system?.adapter_memory_used_gb==null?'—':`${system.adapter_memory_used_gb.toFixed(1)} / ${system.adapter_memory_total_gb?.toFixed(1)} GB`)}</span>
      <span title={ui(`最近一帧解码及预览处理耗时 ${ms(data?.processing_ms)}，换算为处理帧率；不等同于持续播放帧率。`)}>{ui("渲染 ")}{ui(data?.cached_display?'缓存':typeof data?.processing_fps==='number'?`${data.processing_fps.toFixed(1)} FPS`:'—')}</span>
      {ui(data?.playing&&<span title={ui("实际呈现的帧率，包含解码、处理和显示等待；不是素材标称帧率。")}>{ui("显示 ")}{ui(Number(data.presentation_fps||0).toFixed(1))} FPS</span>)}
      {ui((data?.render_pipeline||[]).map((stage:string)=>{
        const measured=effects.find((effect:any)=>effect.stage===stage)
        const nr=stage==='nr'&&!measured
        return <span key={stage} title={ui(nr?'DLSSNR 各层模型推理合计；光流、融合和显示耗时另计。':'最近一次完成的原生调用墙钟耗时；不是 GPU 时间戳。')}>{ui(names[stage.split(':')[0]]||({illustration:'IllustrationJaNai',animesharp:'AnimeSharp'} as Record<string,string>)[stage]||stage)}{ui(nr?' 推理':'')} {ui(stage==='vfg'?'仅导出':stage==='nr'&&data?.inference_reused?'缓存':ms(measured?.process_ms??(nr?(data?.stages?.nr_total_ms??data?.stages?.dlss_inference):undefined)))}</span>
      }))}
      {ui(data?.solver&&<span title={ui("数据一致性融合求解耗时")}>{ui("融合 ")}{ui(ms(data.stages?.hdr_fusion??data.stages?.fusion??data.stages?.fusion_solve??data.solver?.solve_ms))}</span>)}
    </div>
    <button onClick={onQueueClick} className="status-queue">{ui(queueSummary || '队列空闲')}</button></footer>
}
