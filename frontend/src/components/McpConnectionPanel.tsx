import {useEffect, useId, useState} from 'react'
import {Check, Clipboard, ChevronDown, ChevronUp} from 'lucide-react'
import {ui} from '../i18n/locale'
import {fetchMcpConnection, setMcpConnection, type McpConnectionState} from '../services/mcpConnection'
import './mcpConnection.css'

const MCP_TOOLS = [
  ['get_status', '查看工作台与队列状态'],
  ['probe_media', '读取本机素材信息'],
  ['list_presets', '列出已保存预设'],
  ['get_preset', '读取预设参数'],
  ['get_parameters', '查看可调参数'],
  ['resolve_settings', '合并并校验参数'],
  ['save_preset', '保存参数为预设'],
  ['render_video', '提交视频导出'],
  ['render_image', '提交图片导出'],
  ['wait_for_jobs', '等待任务完成或失败'],
  ['get_job', '查询任务状态与结果'],
  ['cancel_job', '取消指定任务'],
  ['start_pending_renders', '继续待处理的 MCP 任务'],
] as const

export function McpConnectionPanel() {
  const detailsId = useId()
  const [state, setState] = useState<McpConnectionState|null>(null)
  const [busy, setBusy] = useState(false)
  const [details, setDetails] = useState(false)
  const [feedback, setFeedback] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    fetchMcpConnection().then(value => {if (active) {setState(value); setError('')}})
      .catch(reason => {if (active) setError((reason as Error).message)})
    return () => {active = false}
  }, [])

  async function toggle(enabled: boolean) {
    setBusy(true); setError(''); setFeedback('')
    try {setState(await setMcpConnection(enabled)); setFeedback(enabled ? 'AI 智能体控制已开启' : 'AI 智能体控制已关闭')}
    catch (reason) {setError((reason as Error).message)}
    finally {setBusy(false)}
  }

  async function copy(text: string, label: string) {
    setFeedback(''); setError('')
    try {await navigator.clipboard.writeText(text); setFeedback(label)}
    catch (reason) {setError((reason as Error).message || '无法访问剪贴板')}
  }

  return <section className="panel mcp-connection-panel">
    <h2>{ui('连接智能体和应用')}</h2>
    <div className="mcp-connection-row">
      <div className="mcp-connection-copy">
        <strong>{ui('允许 AI 智能体控制 Lunarc')}</strong>
      </div>
      <button className="mcp-switch" type="button" role="switch" aria-checked={!!state?.enabled}
        aria-label={ui('允许 AI 智能体控制 Lunarc')} disabled={!state || busy}
        onClick={() => void toggle(!state?.enabled)}>
        <span/>
      </button>
    </div>
    <div className="mcp-connection-row">
      <div className="mcp-connection-copy">
        <strong>{ui('连接智能体')}</strong>
        <span>{ui('将连接提示词复制到智能体对话中。')}</span>
      </div>
      <button className="button" disabled={!state?.enabled || busy} onClick={() => state && void copy(state.prompt, '提示词已复制')}>
        <Clipboard size={15}/>{ui('复制提示词')}
      </button>
    </div>
    <div className="mcp-connection-row mcp-details-row">
      <div className="mcp-connection-copy">
        <strong>{ui('MCP 工具')}</strong>
      </div>
      <button className="button" disabled={!state} aria-expanded={details} aria-controls={detailsId} onClick={() => setDetails(value => !value)}>
        {details ? <ChevronUp size={15}/> : <ChevronDown size={15}/>}{ui(details ? '收起工具' : '查看工具')}
      </button>
    </div>
    {details && state && <div id={detailsId} className="mcp-tools-list">
      {MCP_TOOLS.map(([name, description]) => <div className="mcp-tool-row" key={name}><code>{name}</code><span>{ui(description)}</span></div>)}
    </div>}
    {(error || state?.error || feedback) && <p className={`feedback mcp-feedback ${error || state?.error ? 'is-error' : ''}`} role={error || state?.error ? 'alert' : 'status'}>
      {error || state?.error ? <>{ui('MCP 连接失败')}: {error || state?.error}</> : <><Check size={14}/>{ui(feedback)}</>}
    </p>}
  </section>
}
