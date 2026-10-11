import test from 'node:test'
import assert from 'node:assert/strict'
import { QueueFeed } from '../../frontend/src/services/queueFeed.ts'
import type { QueueState, RenderJob } from '../../frontend/src/types/index.ts'

const job = (id: string, status = 'pending') => ({id, status} as RenderJob)
const state = (...jobs: RenderJob[]): QueueState => ({jobs, active_id: null, queue_running: false})
const flush = async () => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve() }
class Socket {
  onmessage: any = null
  onclose: any = null
  closed = 0
  send(data: object) { this.onmessage?.({data: JSON.stringify(data)}) }
  close() { this.closed++ }
  disconnect() { this.onclose?.() }
}
function harness() {
  const sockets: Socket[] = []
  const requests: {resolve: (value: QueueState) => void; signal: AbortSignal}[] = []
  const feed = new QueueFeed(signal => new Promise(resolve => requests.push({resolve, signal})), () => {
    const socket = new Socket(); sockets.push(socket); return socket
  })
  return {feed, sockets, requests}
}

test('two mounted pages share one connection and read, final subscriber releases resources', async () => {
  const {feed, sockets, requests} = harness()
  let a: QueueState | null = null, b: QueueState | null = null
  const stopA = feed.subscribe(s => { a = s })
  const stopB = feed.subscribe(s => { b = s })
  try {
    assert.equal(sockets.length, 1)
    assert.equal(requests.length, 1)
    requests[0].resolve(state(job('one'))); await flush()
    assert.equal(a, b)
    stopA(); assert.equal(sockets[0].closed, 0)
  } finally { stopA(); stopB() }
  assert.equal(sockets[0].closed, 1)
})

test('late HTTP snapshot and progress cannot recreate a removed task', async () => {
  const {feed, sockets, requests} = harness()
  let latest = state()
  const stop = feed.subscribe(s => {latest = s})
  try {
    sockets[0].send({type: 'init', ...state(job('one'))})
    sockets[0].send({type: 'job_removed', id: 'one'})
    requests[0].resolve(state(job('one'))); await flush()
    sockets[0].send({type: 'job_update', job: job('one', 'completed')})
    assert.deepEqual(latest.jobs, [])
  } finally {stop()}
})

test('ending an older task cannot clear the task currently rendering', () => {
  const {feed, sockets} = harness()
  let latest = state()
  const stop = feed.subscribe(s => {latest = s})
  try {
    sockets[0].send({type: 'init', jobs: [job('old'), job('new', 'running')], active_id: 'new', queue_running: true})
    sockets[0].send({type: 'job_update', job: job('old', 'completed')})
    assert.equal(latest.active_id, 'new')
    for (const status of ['completed', 'failed', 'cancelled', 'skipped']) {
      sockets[0].send({type: 'job_update', job: job('new', 'running')})
      sockets[0].send({type: 'job_update', job: job('new', status)})
      assert.equal(latest.active_id, null)
      assert.equal(latest.queue_running, true)
    }
    sockets[0].send({type: 'queue_finished', ...state(job('new', 'completed'))})
    assert.equal(latest.queue_running, false)
  } finally {stop()}
})

test('connected queue does not poll; disconnection recovers without overlapping reads', async t => {
  t.mock.timers.enable({apis: ['setInterval', 'setTimeout']})
  const {feed, sockets, requests} = harness()
  const stop = feed.subscribe(() => {})
  try {
    requests[0].resolve(state()); await flush()
    sockets[0].send({type: 'init', ...state()})
    t.mock.timers.tick(10000); assert.equal(requests.length, 1)
    sockets[0].disconnect()
    t.mock.timers.tick(2000); assert.equal(requests.length, 2); assert.equal(sockets.length, 2)
    t.mock.timers.tick(2000); assert.equal(requests.length, 2)
    requests[1].resolve(state(job('recovered'))); await flush()
    sockets[1].send({type: 'init', ...state(job('recovered'))})
    t.mock.timers.tick(10000); assert.equal(requests.length, 2)
  } finally {stop()}
})

test('unsubscription aborts pending reads; an old read cannot overwrite a new subscription', async () => {
  const {feed, sockets, requests} = harness()
  const first = feed.subscribe(() => {}); first()
  assert.equal(requests[0].signal.aborted, true)
  let latest = state()
  const second = feed.subscribe(s => {latest = s})
  try {
    sockets[1].send({type: 'init', ...state(job('new'))})
    requests[0].resolve(state(job('obsolete'))); await flush()
    assert.equal(latest.jobs[0].id, 'new')
    assert.equal(requests.length, 2)
  } finally {second()}
})
