import {test} from 'node:test'
import assert from 'node:assert/strict'
import {BackendRecovery,mayRestartBackend} from '../../frontend/electron/backendRecovery.ts'

test('one incident restarts once, preserving the single child boundary',async()=>{
  let starts=0,unblock:()=>void=()=>{}
  const events:any[]=[]
  const recovery=new BackendRecovery(async()=>{starts++;await new Promise<void>(resolve=>{unblock=resolve})},e=>events.push(e))
  const running=recovery.recover('first','device removed')
  await recovery.recover('first','device removed');await recovery.recover('other','device removed')
  assert.equal(starts,1);unblock();await running
  assert.deepEqual(events.map(e=>e.phase),['recovering','restored'])
})
test('repeated fault is blocked within the recovery window',async()=>{
  let now=0,starts=0;const events:any[]=[]
  const recovery=new BackendRecovery(async()=>{starts++},e=>events.push(e),()=>now)
  await recovery.recover('a','reset');now=5000;await recovery.recover('b','reset')
  assert.equal(starts,1);assert.equal(events.at(-1).phase,'blocked')
  now=65000;await recovery.recover('c','reset');assert.equal(starts,2)
})
test('failed start and app shutdown never announce successful recovery',async()=>{
  const events:any[]=[]
  const recovery=new BackendRecovery(async()=>{throw Error('port occupied')},e=>events.push(e))
  await recovery.recover('a','reset');assert.equal(events.at(-1).phase,'blocked')
  recovery.stop();await recovery.recover('b','reset');assert.equal(events.length,2)
})
test('unowned process, healthy backend and failed checkpoint cannot be restarted',()=>{
  const health={service:'lunarc',pid:123,healthy:false,incident_id:'fault',reason:'reset'}
  assert(mayRestartBackend(health,123))
  assert(!mayRestartBackend(health,456));assert(!mayRestartBackend(health,null))
  assert(!mayRestartBackend({...health,healthy:true},123))
  assert(!mayRestartBackend({...health,queue_storage_error:'disk full'},123))
})
