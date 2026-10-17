// localized-render-sinks
import {ui} from '../i18n/locale'
import type {SystemInfo} from '../types'
import {GpuMemoryMeter} from './GpuMemoryMeter'

const gib=(bytes:number|undefined)=>bytes==null?'—':`${(bytes/1024**3).toFixed(2)} GiB`
export function GpuMemoryPanel({memory}:{memory:SystemInfo['gpu_memory']}){
  return <section className="panel"><h2>{ui("显存分配")}</h2>{ui(!memory?<p>{ui("处理第一帧后显示显存统计。")}</p>:<>
    <p>{ui(memory.phase==='export'?'导出':'预览')} · {ui(memory.active?'最近采样':'上次采样')} · {ui(new Date(memory.sampled_at*1000).toLocaleTimeString())}</p>
    <GpuMemoryMeter used={memory.process?.local_usage==null?undefined:memory.process.local_usage/1024**3}
      total={memory.process?.local_budget==null?undefined:memory.process.local_budget/1024**3} label="进程用量 / 系统预算" unit="GiB"/>
    <dl className="fact-list"><dt>{ui("进程用量 / 系统预算")}</dt><dd>{ui(gib(memory.process?.local_usage))} / {ui(gib(memory.process?.local_budget))}</dd>
      <dt>{ui("进程非本地内存")}</dt><dd>{ui(gib(memory.process?.nonlocal_usage))}</dd>
      <dt>{ui("已统计设备资源分配")}</dt><dd>{ui(gib(memory.known_device_allocations_bytes))}</dd>
      {ui(memory.preview_cache&&<><dt>{ui("预览缓存（逻辑容量）")}</dt><dd>{ui(gib(memory.preview_cache.private_plane_bytes))}</dd></>)}
      {ui(memory.torch&&<><dt>{ui("PyTorch 张量 / 保留缓存")}</dt><dd>{ui(gib(memory.torch.allocated_bytes))} / {ui(gib(memory.torch.reserved_bytes))}</dd></>)}
    </dl>
    <details><summary>{ui("各阶段资源")}</summary><table className="memory-table"><thead><tr><th>{ui("资源")}</th><th>{ui("设备分配")}</th><th>{ui("CPU 暂存")}</th></tr></thead><tbody>{ui(memory.components?.map(row=><tr key={row.owner} title={ui(row.scope)}><td>{ui(row.kind)}</td><td>{ui(gib(row.allocated_device_bytes))}</td><td>{ui(row.host_staging_bytes==null?'—':gib(row.host_staging_bytes))}</td></tr>))}</tbody></table></details>
    <p>{ui("资源分配量与实际显存用量口径不同，不能相加。SDK 内部模型、工作区和编解码帧池尚不能逐项归属；帧间采样不包含所有瞬时峰值。")}</p>
    {ui(memory.error&&<p role="status">{ui("统计暂不可用：")}{ui(memory.error)}</p>)}
  </>)}</section>
}
