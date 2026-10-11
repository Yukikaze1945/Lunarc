import {useEffect, useState} from 'react'
import {RefreshCw} from 'lucide-react'
import type {EnvironmentReport} from '../types'
import {fetchEnvironment} from '../services/api'

const labels = {ok: '检查通过', pending: '依赖齐全', missing: '缺少组件', error: '检测失败'}
export function EnvironmentPanel() {
  const [report, setReport] = useState<EnvironmentReport | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [startup, setStartup] = useState<{message: string; logPath?: string} | null>(null)
  async function check(refresh: boolean) {
    setBusy(true); setError('')
    try {setReport(await fetchEnvironment(refresh)); setStartup(null)}
    catch (e: any) {
      setError(e.message)
      try {setStartup(await window.electronAPI?.getBackendStartup?.() || null)} catch {}
    } finally {setBusy(false)}
  }
  useEffect(() => {void check(false)}, [])
  return <section className="panel environment-panel">
    <div className="environment-heading"><h2>运行环境</h2><button className="button" disabled={busy} onClick={() => void check(true)}><RefreshCw size={15}/>{busy ? '检测中…' : '重新检测'}</button></div>
    {error && <p role="alert">{startup?.message || error}{startup?.logPath && <><br/>日志：{startup.logPath}</>}</p>}
    {report && <>
      <p>{report.standalone && report.isolated ? '独立运行环境' : '开发运行环境'} · {new Date(report.checked_at * 1000).toLocaleTimeString()} 检测</p>
      <div className="environment-features">{report.features.map(check => <div className="environment-row" key={check.id}><strong>{check.label}</strong><span className={`environment-status ${check.status}`}>{labels[check.status]}</span><span>{check.detail}</span></div>)}</div>
      <details><summary>组件详情</summary>{report.checks.map(check => <div className="environment-check" key={check.id}><div><strong>{check.label}</strong><span className={`environment-status ${check.status}`}>{labels[check.status]}</span></div><p>{check.detail}</p>{check.status !== 'ok' && check.action && <p>{check.action}</p>}</div>)}</details>
    </>}
    <p role="status">{busy ? '正在检查驱动、CUDA、运行库与模型…' : report?.features.length ? '依赖检查不代替实际运行验证。' : ''}</p>
  </section>
}
