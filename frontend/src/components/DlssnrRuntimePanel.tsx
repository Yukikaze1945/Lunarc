import {useEffect, useRef, useState} from 'react'
import {FolderOpen, RefreshCw, Trash2} from 'lucide-react'
import {runtimeRequest, type DlssnrStatus, type DlssnrVersion} from '../services/api'
import './dlssnrRuntime.css'

const phases: Record<string, string> = {copying: '复制中', downloading: '下载中', verifying: '校验中', extracting: '解包中'}

export function DlssnrRuntimePanel() {
  const [state, setState] = useState<DlssnrStatus | null>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [path, setPath] = useState('')
  const [kind, setKind] = useState<'directory' | 'download'>('directory')
  const [location, setLocation] = useState('')
  const [label, setLabel] = useState('')
  const [sha, setSha] = useState('')
  const [confirmed, setConfirmed] = useState(false)
  const [archiveSha, setArchiveSha] = useState('')
  const [communityApproved, setCommunityApproved] = useState(false)
  const [experimental, setExperimental] = useState(false)
  const refreshRevision = useRef(0)
  const refresh = async () => {
    const revision = ++refreshRevision.current
    const result = await runtimeRequest<DlssnrStatus>()
    if (revision === refreshRevision.current) setState(result)
  }
  useEffect(() => {void refresh().catch(e => setError(e.message))}, [])
  useEffect(() => {
    if (!busy) return
    const timer = window.setInterval(() => {void refresh().catch(() => {})}, 2000)
    return () => window.clearInterval(timer)
  }, [busy])
  async function run(action: () => Promise<unknown>, success: string) {
    if (busy) return
    setBusy(true); setError(''); setMessage('')
    let completed = false
    try {await action(); completed = true}
    catch (e: any) {setError(e.message)}
    finally {try {await refresh(); if (completed) setMessage(success)} catch (e: any) {setError(e.message)} setBusy(false)}
  }
  async function importPath(value: string) {
    if (value) await run(() => runtimeRequest<DlssnrVersion>('/import', 'POST', {path: value}), '已校验并缓存，请选择使用版本。')
  }
  async function chooseFile() {
    try {
      const value = await window.electronAPI?.openRuntimeDialog?.()
      if (value) {setPath(value); await importPath(value)}
    } catch (e: any) {setError(e.message)}
  }
  const operation = state?.operation
  return <section className="panel dlssnr-runtime" onDragOver={e => {e.preventDefault(); e.stopPropagation()}} onDrop={e => {
    e.preventDefault(); e.stopPropagation()
    if (busy) return
    const file = e.dataTransfer.files[0]
    if (!file || !/\.(dll|zip)$/i.test(file.name)) {setError('请拖入 DLSSNR DLL 或 ZIP 包'); return}
    const value = window.electronAPI?.getPathForFile(file)
    if (value) {setPath(value); void importPath(value)} else setError('无法取得文件路径，请选择文件或粘贴本机路径')
  }}>
    <div className="environment-heading"><h2>DLSSNR 运行库</h2><button className="button" disabled={busy} onClick={() => void run(refresh, '')}><RefreshCw size={15}/>刷新</button></div>
    <div className="runtime-current"><strong>{state?.active ? `版本 ${state.active.version}` : '尚未选择'}</strong><span>{state?.active && !state.error ? 'NVIDIA 签名有效 · SHA-256 已校验' : state?.error?.message}</span></div>
    <div className="runtime-import">
      <input aria-label="DLSSNR DLL 或 ZIP 路径" value={path} onChange={e => setPath(e.target.value)} placeholder="拖入 DLL / ZIP，或粘贴本机路径" disabled={busy}/>
      {window.electronAPI?.openRuntimeDialog && <button className="button" disabled={busy} onClick={() => void chooseFile()}><FolderOpen size={15}/>选择文件</button>}
      <button className="button" disabled={busy || !path.trim()} onClick={() => void importPath(path.trim())}>导入</button>
    </div>
    {state?.legacy_path && !state.versions.length && <button className="button quiet" disabled={busy} onClick={() => void importPath(state.legacy_path!)}>导入本机已有运行库</button>}
    <div className="runtime-versions">{state?.versions.map(v => <div className="runtime-version" key={v.id}>
      <div><strong>{v.version}</strong><code title={v.sha256}>{v.sha256.slice(0, 12)}…</code><span>{v.compatibility === 'known' ? '已核对版本' : '未验收版本'}</span></div>
      <div className="runtime-actions"><button className="button" disabled={busy || state.active?.id === v.id && !state.error || v.compatibility !== 'known' && !experimental} onClick={() => void run(() => runtimeRequest('/activate', 'POST', {id: v.id, allow_unrecognized: experimental}), '已选择，新会话使用该版本。')}>{state.active?.id === v.id ? '已选择' : '使用'}</button><button className="button quiet" disabled={busy || state.active?.id === v.id} aria-label={`删除版本 ${v.version}`} onClick={() => void run(() => runtimeRequest(`/versions/${v.id}`, 'DELETE'), '缓存版本已删除。')}><Trash2 size={15}/></button></div>
    </div>)}</div>
    {state?.active && <div className="runtime-actions"><button className="button quiet" disabled={busy} onClick={() => void run(() => runtimeRequest('/verify', 'POST', {id: state.active!.id}), '文件哈希与数字签名检查通过。')}>重新校验</button><button className="button quiet" disabled={busy} onClick={() => void run(() => runtimeRequest('/activate', 'POST', {id: null}), '已取消选择。')}>取消选择</button></div>}
    <details><summary>版本目录与下载</summary>
      <label className="runtime-consent"><input type="checkbox" checked={communityApproved} disabled={busy} onChange={e => setCommunityApproved(e.target.checked)}/>我确认使用 Feeder 所用的社区来源，并有权取得和使用运行库</label>
      {state?.catalog.map(v => <div className="runtime-version" key={v.id}><div><strong>{v.version}</strong><span>{v.source}</span></div><button className="button" disabled={busy || !communityApproved || v.downloaded} onClick={() => void run(() => runtimeRequest('/import', 'POST', {catalog_id: v.id, confirmed: communityApproved}), '已下载、解包并校验，请选择使用版本。')}>{v.downloaded ? '已缓存' : '下载'}</button></div>)}
    </details>
    <details><summary>可信来源</summary>
      {state?.sources.map(source => <div className="runtime-source" key={source.id}><div><strong>{source.label}</strong><span title={source.location}>{source.location}</span></div><div className="runtime-actions"><button className="button" disabled={busy} onClick={() => void run(() => runtimeRequest('/import', 'POST', {source_id: source.id}), '已校验并缓存，请选择使用版本。')}>{source.kind === 'download' ? '下载并校验' : '导入'}</button><button className="button quiet" disabled={busy} aria-label={`移除来源 ${source.label}`} onClick={() => void run(() => runtimeRequest(`/sources/${source.id}`, 'DELETE'), '来源已移除。')}><Trash2 size={15}/></button></div></div>)}
      <fieldset disabled={busy} className="runtime-source-form"><legend>添加来源</legend>
        <label>类型<select value={kind} onChange={e => {setKind(e.target.value as typeof kind); setLocation(''); setConfirmed(false)}}><option value="directory">本机目录</option><option value="download">HTTPS 下载</option></select></label>
        <label>名称<input value={label} maxLength={120} onChange={e => setLabel(e.target.value)} placeholder="来源名称"/></label>
        <label className="runtime-wide">{kind === 'download' ? 'DLL 下载地址' : '包含 DLL 的目录'}<div className="runtime-import"><input value={location} onChange={e => {setLocation(e.target.value); setConfirmed(false)}} placeholder={kind === 'download' ? 'https://…/nvngx_dlssnr.dll' : '目录路径'}/>{kind === 'directory' && window.electronAPI?.openRuntimeDirDialog && <button type="button" className="button" onClick={async () => {try {const value = await window.electronAPI?.openRuntimeDirDialog?.(); if (value) {setLocation(value); setConfirmed(false)}} catch (e: any) {setError(e.message)}}}>选择目录</button>}</div></label>
        <label className="runtime-wide">SHA-256{kind === 'directory' ? '（可选）' : ''}<input value={sha} onChange={e => {setSha(e.target.value); setConfirmed(false)}} placeholder="64 位十六进制哈希" maxLength={64}/></label>
        {kind === 'download' && <label className="runtime-wide">ZIP SHA-256（ZIP 下载时填写）<input value={archiveSha} onChange={e => {setArchiveSha(e.target.value); setConfirmed(false)}} maxLength={64}/></label>}
        <label className="runtime-wide runtime-consent"><input type="checkbox" checked={confirmed} onChange={e => setConfirmed(e.target.checked)}/>我信任此来源，并确认有权取得和使用该运行库</label>
        <button className="button" disabled={!confirmed || !label.trim() || !location.trim() || (kind === 'download' && !/^[a-f\d]{64}$/i.test(sha))} onClick={() => void run(async () => {await runtimeRequest('/sources', 'POST', {kind, location: location.trim(), label, sha256: sha, archive_sha256: archiveSha, confirmed}); setLocation(''); setLabel(''); setSha(''); setArchiveSha(''); setConfirmed(false)}, '可信来源已保存。')}>添加来源</button>
      </fieldset>
      <p>下载需提供 DLL 哈希；ZIP 还需包哈希。目录内导入 nvngx_dlssnr.dll。</p>
    </details>
    {state?.versions.some(v => v.compatibility !== 'known') && <details><summary>未验收版本</summary><label className="runtime-consent"><input type="checkbox" checked={experimental} disabled={busy} onChange={e => setExperimental(e.target.checked)}/>允许实验性使用签名有效的未验收版本</label><p>仍校验 NVIDIA 签名与固定哈希；模型接口和画质尚未验收。</p></details>}
    <details><summary>缓存与许可</summary><p className="runtime-path">{state?.cache_dir}</p>{state?.active?.path && <p className="runtime-path">当前文件：{state.active.path}</p>}{state?.active && <p className="runtime-path">SHA-256：{state.active.sha256}</p>}<p>运行库由用户提供，仍受 NVIDIA 原许可约束。签名校验不代表再分发授权或设备兼容性。</p></details>
    {busy && <div className="runtime-actions"><p role="status">{phases[operation?.phase || ''] || '处理中'}{operation?.total ? ` · ${Math.round(operation.received / operation.total * 100)}%` : '…'}</p><button className="button quiet" onClick={() => void runtimeRequest('/cancel', 'POST').catch(e => setError(e.message))}>取消</button></div>}
    {error ? <p role="alert" className="runtime-error">{error}</p> : message && <p role="status">{message}</p>}
  </section>
}
