import type { QueueState, RenderJob } from '../types/index.ts'

type Listener = (state: QueueState) => void
type Socket = Pick<WebSocket, 'onmessage' | 'onclose' | 'close'>
const EMPTY: QueueState = {jobs: [], active_id: null, queue_running: false}

// Queue messages and HTTP recovery share one state and one ordering boundary.
export class QueueFeed {
  private listeners = new Set<Listener>()
  private state: QueueState | null = null
  private socket: Socket | null = null
  private connected = false
  private revision = 0
  private pending: Promise<QueueState | null> | null = null
  private controller: AbortController | null = null
  private reconnect: ReturnType<typeof setTimeout> | null = null
  private recovery: ReturnType<typeof setInterval> | null = null
  private read: (signal: AbortSignal) => Promise<QueueState>
  private open: () => Socket

  constructor(read: (signal: AbortSignal) => Promise<QueueState>, open: () => Socket) {
    this.read = read
    this.open = open
  }

  getSnapshot(): QueueState | null { return this.state }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener)
    if (this.state) listener(this.state)
    if (this.listeners.size === 1) {
      this.connect()
      void this.refresh()
      this.recovery = setInterval(() => {
        if (!this.connected) void this.refresh()
      }, 2000)
    }
    return () => {
      this.listeners.delete(listener)
      if (this.listeners.size) return
      this.revision++
      this.controller?.abort()
      this.controller = null
      // An aborted read from the old subscription cannot own a new subscription.
      this.pending = null
      if (this.recovery) clearInterval(this.recovery)
      if (this.reconnect) clearTimeout(this.reconnect)
      this.recovery = this.reconnect = null
      if (this.socket) {
        this.socket.onmessage = this.socket.onclose = null
        this.socket.close()
      }
      this.socket = null
      this.connected = false
      this.state = null
    }
  }

  private publish(state: QueueState) {
    this.state = state
    for (const listener of this.listeners) {
      try { listener(state) } catch (error) { console.error('Queue subscriber failed', error) }
    }
  }

  refresh(): Promise<QueueState | null> {
    if (this.pending) return this.pending
    const revision = this.revision
    const controller = new AbortController()
    this.controller = controller
    const timeout = setTimeout(() => controller.abort(), 5000)
    controller.signal.addEventListener('abort', () => clearTimeout(timeout), {once: true})
    const request = this.read(controller.signal).then(state => {
      // Progress/removal received after this request began is newer than its snapshot.
      if (this.listeners.size && this.revision === revision) this.publish(state)
      return this.state
    }).catch(() => this.state).finally(() => {
      clearTimeout(timeout)
      if (this.pending === request) this.pending = null
      if (this.controller === controller) this.controller = null
    })
    this.pending = request
    return request
  }

  private connect() {
    if (!this.listeners.size) return
    const retry = () => {
      this.connected = false
      if (this.listeners.size) this.reconnect = setTimeout(() => this.connect(), 2000)
    }
    try {
      const socket = this.open()
      this.socket = socket
      socket.onclose = () => {
        if (this.socket !== socket) return
        this.socket = null
        retry()
      }
      socket.onmessage = event => {
        if (this.socket !== socket) return
        try { this.receive(JSON.parse(event.data)) } catch { /* Recovery read supplies valid state. */ }
      }
    } catch { retry() }
  }

  private receive(data: any) {
    const previous = this.state || EMPTY
    let jobs = previous.jobs
    let active = previous.active_id
    let running = typeof data.queue_running === 'boolean' ? data.queue_running : previous.queue_running
    if (['init', 'job_added', 'job_removed', 'job_retried', 'queue_started', 'queue_finished'].includes(data.type)) {
      if (Array.isArray(data.jobs)) jobs = data.jobs
      else if (data.type === 'job_added' && data.job && !jobs.some(j => j.id === data.job.id)) jobs = [...jobs, data.job]
      else if (data.type === 'job_removed') jobs = jobs.filter(j => j.id !== data.id)
      if ('active_id' in data) active = data.active_id || null
      if (data.type === 'job_removed' && active === data.id) active = null
      if (data.type === 'queue_finished') { active = null; running = false }
    } else if (data.type === 'job_update' && data.job) {
      const updated: RenderJob = data.job
      // Delayed progress events never recreate a row that the operator removed.
      if (!jobs.some(j => j.id === updated.id)) return
      jobs = jobs.map(j => j.id === updated.id ? updated : j)
      if (updated.status === 'running') active = updated.id
      else if (active === updated.id && ['completed', 'failed', 'cancelled', 'skipped'].includes(updated.status)) active = null
    } else return
    this.revision++
    if (data.type === 'init') this.connected = true
    this.publish({jobs, active_id: active, queue_running: running})
  }
}
