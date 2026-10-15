import { SystemInfo, VideoInfo, DLSSSettings, RenderJob, QueueState, EnvironmentReport } from '../types'

const API_BASE = 'http://127.0.0.1:18923'

export interface PreviewMemoryState {suspended:boolean;releasing:boolean;error?:string|null;report?:unknown}
export async function previewMemory(action?:'release'|'resume'):Promise<PreviewMemoryState> {
  const response=await fetch(`${API_BASE}/api/preview/memory${action?'/'+action:''}`,{
    method:action?'POST':'GET',headers:authHeaders()})
  const data=await response.json()
  if(!response.ok)throw new Error(data.detail||'预览显存操作失败')
  return data
}

export async function imageModelRequest(action='',body?:{key:string;path?:string}) {
  const response=await fetch(`${API_BASE}/api/image/models${action}`,{
    method:body?'POST':'GET',headers:authHeaders({'Content-Type':'application/json'}),
    ...(body?{body:JSON.stringify(body)}:{})})
  const data=await response.json()
  if(!response.ok)throw new Error(data.detail||'图片模型操作失败')
  return data
}

// Local-backend session token (P0-01): injected by Electron preload; plain
// browser dev can set localStorage 'dlss_local_token' to match the backend.
export function localToken(): string {
  try {
    return window.electronAPI?.localToken
      || localStorage.getItem('dlss_local_token')
      || ''
  } catch {
    return ''
  }
}

function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const t = localToken()
  return t ? { 'X-DLSS-Token': t, ...extra } : { ...extra }
}

export function wsUrl(path: 'player' | 'queue'): string {
  const t = localToken()
  const base = `ws://127.0.0.1:18923/ws/${path}`
  return t ? `${base}?token=${encodeURIComponent(t)}` : base
}

export async function fetchSystemInfo(): Promise<SystemInfo> {
  const res = await fetch(`${API_BASE}/api/system`, { headers: authHeaders() })
  if (!res.ok) throw new Error('Failed to fetch system info')
  const system=await res.json()
  try{
    const memory=await fetch(`${API_BASE}/api/gpu/memory`,{headers:authHeaders(),signal:AbortSignal.timeout(2000)})
    if(memory.ok)system.gpu_memory=(await memory.json()).memory
  }catch{/* Startup/older backend can still report system information. */}
  return system
}

export async function fetchEnvironment(refresh = false): Promise<EnvironmentReport> {
  const res = await fetch(`${API_BASE}/api/environment?refresh=${refresh}`, {headers: authHeaders()})
  if (!res.ok) throw new Error('无法获取环境检测结果')
  return res.json()
}

export interface DlssnrVersion {
  id: string; version: string; sha256: string; signature: string; bytes: number
  path?: string; origin?: {kind: string; label?: string}
  compatibility: 'known' | 'uncharacterized'
}
export interface DlssnrSource {
  id: string; kind: 'directory' | 'download'; label: string; location: string; sha256: string
}
export interface DlssnrStatus {
  active: DlssnrVersion | null; versions: DlssnrVersion[]; sources: DlssnrSource[]
  error: {code: string; message: string} | null; cache_dir: string; legacy_path?: string | null
  operation?: {phase: string; received: number; total: number | null} | null
  catalog: {id: string; version: string; source: string; downloaded: boolean}[]
}
export async function runtimeRequest<T>(path = '', method = 'GET', body?: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}/api/runtime/dlssnr${path}`, {
    method, headers: authHeaders({'Content-Type': 'application/json'}),
    ...(body === undefined ? {} : {body: JSON.stringify(body)}),
  })
  const data = await response.json()
  if (!response.ok) throw new Error(data.detail?.message || data.detail || '运行库操作失败')
  return data as T
}

export async function openNativeFileDialog(): Promise<string | null> {
  if (window.electronAPI && typeof window.electronAPI.openFileDialog === 'function') {
    try {
      const p = await window.electronAPI.openFileDialog()
      if (p) return p
    } catch (e) {
      console.warn('Electron openFileDialog error:', e)
    }
  }
  try {
    const res = await fetch(`${API_BASE}/api/dialog/open_file`, { method: 'POST', headers: authHeaders() })
    if (res.ok) {
      const data = await res.json()
      if (data.path) return data.path
    }
  } catch (e) {
    console.warn('Backend dialog error:', e)
  }
  return null
}

export async function openNativeDirDialog(): Promise<string | null> {
  if (window.electronAPI && typeof window.electronAPI.openDirDialog === 'function') {
    try {
      const p = await window.electronAPI.openDirDialog()
      if (p) return p
    } catch (e) {
      console.warn('Electron openDirDialog error:', e)
    }
  }
  try {
    const res = await fetch(`${API_BASE}/api/dialog/open_dir`, { method: 'POST', headers: authHeaders() })
    if (res.ok) {
      const data = await res.json()
      if (data.path) return data.path
    }
  } catch (e) {
    console.warn('Backend dir dialog error:', e)
  }
  return null
}

export async function loadVideoFile(path: string): Promise<VideoInfo> {
  const res = await fetch(`${API_BASE}/api/video/load`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ path }),
  })
  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Failed to load video')
  }
  const data = await res.json()
  return data.video
}

export async function exportImage(request: {path:string;output_path:string;format:string;quality:number;preserve_alpha:boolean;overwrite:boolean}, settings:DLSSSettings): Promise<any> {
  await saveSettings(settings)
  const res=await fetch(`${API_BASE}/api/image/export`,{method:'POST',headers:authHeaders({'Content-Type':'application/json'}),body:JSON.stringify(request)})
  const data=await res.json()
  if(!res.ok)throw new Error(data.detail||'图片导出失败')
  return data.image
}

export async function imageExportFormats():Promise<Record<string,boolean>> {
  const res=await fetch(`${API_BASE}/api/image/formats`,{headers:authHeaders()})
  if(!res.ok)throw new Error('无法检查图片编码器')
  return (await res.json()).formats
}

export async function imageThumbnail(path:string,edge=720): Promise<Blob> {
  const res=await fetch(`${API_BASE}/api/image/thumbnail?path=${encodeURIComponent(path)}&edge=${edge}`,{headers:authHeaders()})
  if(!res.ok)throw new Error('无法读取图片缩略图')
  return res.blob()
}

export async function fetchCurrentVideo(): Promise<VideoInfo | null> {
  const res = await fetch(`${API_BASE}/api/video/current`, { headers: authHeaders() })
  if (!res.ok) return null
  const data = await res.json()
  return data.video
}

export async function fetchSettings(): Promise<DLSSSettings> {
  const res = await fetch(`${API_BASE}/api/settings`, { headers: authHeaders() })
  if (!res.ok) throw new Error('Failed to fetch settings')
  return res.json()
}

export async function fetchFlowProviders(): Promise<{ providers: any[]; nvof_probe: any; provider_contracts?: Record<string, unknown> }> {
  const res = await fetch(`${API_BASE}/api/flow/providers`, { headers: authHeaders() })
  if (!res.ok) throw new Error('Failed to fetch flow providers')
  return res.json()
}

async function writeSettings(settings: Partial<DLSSSettings>, inputAt: number): Promise<DLSSSettings> {
  const started = performance.now()
  const res = await fetch(`${API_BASE}/api/settings`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(settings),
  })
  if (!res.ok) {
    // The backend's own reason is what the operator can act on: an experimental HDR
    // write is refused with the clause that refused it, and "Failed to update settings"
    // would hide whether it was the flag, the paper white or the host that was named.
    const detail = await res.json().catch(() => null)
    throw new Error(typeof detail?.detail === 'string'
      ? detail.detail : 'Failed to update settings')
  }
  const data = await res.json()
  window.dispatchEvent(new CustomEvent('lunarc:settings-applied', {detail:{
    revision:data.preview_revision, inputAt, settings_apply_ms:performance.now()-inputAt,
    settings_http_ms:performance.now()-started}}))
  return data.settings
}

let settingsInFlight = false
let pendingSettings: Partial<DLSSSettings> = {}
let settingsWaiters: {resolve: (s: DLSSSettings) => void; reject: (e: unknown) => void}[] = []
let latestSettingsInput = 0

export function saveSettings(settings: Partial<DLSSSettings>): Promise<DLSSSettings> {
  // One request plus one merged pending patch. Slider input cannot create an
  // arbitrarily long HTTP queue or let older patches overtake newer ones.
  pendingSettings = {...pendingSettings, ...settings}
  latestSettingsInput = performance.now()
  window.dispatchEvent(new Event('lunarc:settings-pending'))
  const result = new Promise<DLSSSettings>((resolve, reject) => settingsWaiters.push({resolve, reject}))
  if (!settingsInFlight) {
    settingsInFlight = true
    queueMicrotask(async () => {
      try {
        while (settingsWaiters.length) {
          const patch = pendingSettings, waiters = settingsWaiters, inputAt = latestSettingsInput
          pendingSettings = {}; settingsWaiters = []
          try {
            const saved = await writeSettings(patch, inputAt)
            waiters.forEach(waiter => waiter.resolve(saved))
          } catch (error) {waiters.forEach(waiter => waiter.reject(error))}
        }
      } finally {
        settingsInFlight = false
        window.dispatchEvent(new Event('lunarc:settings-settled'))
      }
    })
  }
  return result
}

export function buildFrameUrl(
  frame: number,
  view: string = 'dlss',
  splitX: number = 0.5,
  zoom: number = 1.0,
  panX: number = 0.5,
  panY: number = 0.5
): string {
  const params = new URLSearchParams({
    frame: frame.toString(),
    view,
    split_x: splitX.toFixed(3),
    zoom: zoom.toFixed(2),
    pan_x: panX.toFixed(3),
    pan_y: panY.toFixed(3),
    t: Date.now().toString(),
  })
  const t = localToken()
  if (t) params.set('token', t)
  return `${API_BASE}/api/frame?${params.toString()}`
}

export async function fetchQueue(signal?: AbortSignal): Promise<QueueState> {
  const res = await fetch(`${API_BASE}/api/queue`, { headers: authHeaders(), signal })
  if (!res.ok) throw new Error('Failed to fetch queue')
  const data = await res.json()
  return {
    jobs: data.jobs,
    active_id: data.active_id ?? null,
    // A client that joined a run in flight has to see it locked; treating an absent
    // flag as false would offer a retry the backend is about to refuse.
    queue_running: Boolean(data.queue_running),
  }
}

export async function addToQueue(job: Partial<RenderJob>): Promise<RenderJob> {
  const res = await fetch(`${API_BASE}/api/queue/add`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(job),
  })
  if (!res.ok) {
    // The service refuses unrunnable jobs with the reason in `detail`; without
    // reading it the opt-in rejection arrives as an undifferentiated failure.
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to add job to queue')
  }
  const data = await res.json()
  return data.job
}

export async function removeFromQueue(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/queue/remove`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ id }),
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({}))
    throw new Error(error.detail || '删除任务失败')
  }
}

// The only route by which a finished job renders again. The queue runs pending jobs
// and nothing else, so a failure stays put until this is called on it by name.
export async function retryQueueJob(id: string): Promise<QueueState> {
  const res = await fetch(`${API_BASE}/api/queue/retry`, {
    method: 'POST',
    headers: authHeaders({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ id }),
  })
  if (!res.ok) {
    // 409 means a round is still in flight, and the reason is the actionable part:
    // without reading it the operator sees "Failed to retry" and no way to act on it.
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to retry job')
  }
  const data = await res.json()
  return {
    jobs: data.jobs,
    active_id: data.active_id ?? null,
    queue_running: Boolean(data.queue_running),
  }
}

export async function startQueue(): Promise<QueueState | null> {
  const res = await fetch(`${API_BASE}/api/queue/start`, { method: 'POST', headers: authHeaders() })
  if (!res.ok) return null
  // "already_running" answers with the same state, so a second client pressing render
  // learns the queue is locked instead of assuming its own request started a round.
  const data = await res.json().catch(() => null)
  if (!data || !Array.isArray(data.jobs)) return null
  return { jobs: data.jobs, active_id: data.active_id ?? null, queue_running: Boolean(data.queue_running) }
}

export async function cancelQueue(): Promise<void> {
  const res = await fetch(`${API_BASE}/api/queue/cancel`, { method: 'POST', headers: authHeaders() })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || '取消任务失败')
  }
}

export async function fetchDiagnosticLogs(limit: number = 200, signal?: AbortSignal): Promise<{ logs: any[] }> {
  const res = await fetch(`${API_BASE}/api/diagnostics/logs?limit=${limit}`, { headers: authHeaders(), signal })
  if (!res.ok) throw new Error('Failed to fetch diagnostic logs')
  return res.json()
}

export async function fetchJobDiagnostics(jobId: string, summary = false): Promise<any> {
  const res = await fetch(`${API_BASE}/api/diagnostics/job/${jobId}?summary=${summary}`, { headers: authHeaders() })
  if (!res.ok) throw new Error('Failed to fetch job diagnostics')
  return res.json()
}
