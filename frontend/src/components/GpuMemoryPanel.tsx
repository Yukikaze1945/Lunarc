import type {SystemInfo} from '../types'

const gib=(bytes:number|undefined)=>bytes==null?'—':`${(bytes/1024**3).toFixed(2)} GiB`
export function GpuMemoryPanel({memory}:{memory:SystemInfo['gpu_memory']}){
  return <section className="panel"><h2>显存分配</h2>{!memory?<p>处理第一帧后显示显存统计。</p>:<>
    <p>{memory.phase==='export'?'导出':'预览'} · {memory.active?'最近采样':'上次采样'} · {new Date(memory.sampled_at*1000).toLocaleTimeString()}</p>
    <dl className="fact-list"><dt>进程用量 / 系统预算</dt><dd>{gib(memory.process?.local_usage)} / {gib(memory.process?.local_budget)}</dd>
      <dt>进程非本地内存</dt><dd>{gib(memory.process?.nonlocal_usage)}</dd>
      <dt>已统计设备资源分配</dt><dd>{gib(memory.known_device_allocations_bytes)}</dd>
      {memory.preview_cache&&<><dt>预览缓存（逻辑容量）</dt><dd>{gib(memory.preview_cache.private_plane_bytes)}</dd></>}
      {memory.torch&&<><dt>PyTorch 张量 / 保留缓存</dt><dd>{gib(memory.torch.allocated_bytes)} / {gib(memory.torch.reserved_bytes)}</dd></>}
    </dl>
    <details><summary>各阶段资源</summary><table className="memory-table"><thead><tr><th>资源</th><th>设备分配</th><th>CPU 暂存</th></tr></thead><tbody>{memory.components?.map(row=><tr key={row.owner} title={row.scope}><td>{row.kind}</td><td>{gib(row.allocated_device_bytes)}</td><td>{row.host_staging_bytes==null?'—':gib(row.host_staging_bytes)}</td></tr>)}</tbody></table></details>
    <p>资源分配量与实际显存用量口径不同，不能相加。SDK 内部模型、工作区和编解码帧池尚不能逐项归属；帧间采样不包含所有瞬时峰值。</p>
    {memory.error&&<p role="status">统计暂不可用：{memory.error}</p>}
  </>}</section>
}
