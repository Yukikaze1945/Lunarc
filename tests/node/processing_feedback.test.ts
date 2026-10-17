import {test} from 'node:test'
import assert from 'node:assert/strict'
import {gpuFailure,memoryLevel} from '../../frontend/src/services/processingFeedback.ts'

test('explicit allocation failures share one message and retain raw details',()=>{
  for(const raw of ['CUDA_ERROR_OUT_OF_MEMORY','CUDA out of memory','CUDA error: out of memory','E_OUTOFMEMORY 0x8007000e','nvenc: NV_ENC_ERR_OUT_OF_MEMORY','cuMemAlloc input failed (2)','后端原生层已熔断（GPU 内存分配失败）']){
    const result=gpuFailure(raw)
    assert.equal(result?.kind,'memory',raw)
    assert.equal(result?.detail,raw)
    assert.ok(result?.message.startsWith('显存不足'))
  }
})
test('device resets are not asserted to be out of memory',()=>{
  assert.equal(gpuFailure('map readback: 0x887a0005 (removed reason 0x887a0006)')?.kind,'reset')
  for(const raw of ['tune: Invalid argument (-22)','CUDA context changed','host already active','access violation','AVIF metadata mismatch'])assert.equal(gpuFailure(raw),null)
})
test('memory bar thresholds handle missing values and budget oversubscription',()=>{
  assert.equal(memoryLevel(79,100)?.level,'normal')
  assert.equal(memoryLevel(80,100)?.level,'warning')
  assert.equal(memoryLevel(92,100)?.level,'critical')
  assert.deepEqual(memoryLevel(120,100),{percent:120,fill:100,level:'critical'})
  for(const [used,total] of [[1,0],[-1,16],[NaN,16],[1,Infinity],[undefined,16]])assert.equal(memoryLevel(used,total),null)
})
