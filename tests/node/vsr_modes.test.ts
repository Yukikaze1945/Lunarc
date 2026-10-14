import test from 'node:test'
import assert from 'node:assert/strict'
import {vsrSameSize,vsrModePatch} from '../../frontend/src/services/vsrModes.ts'
import {vsrOutputSize,nativeOutputBlockers} from '../../frontend/src/services/renderPipeline.ts'
import {presetSettings,presetValue} from '../../frontend/src/services/presets.ts'

test('same-size modes do not erase upscale settings or reject 1x',()=>{
  const s=presetSettings({render_pipeline:['vsr'],vsr_mode:'denoise',vsr_output_mode:'scale',vsr_scale:4})
  assert.deepEqual(vsrOutputSize(s,{width:640,height:360}),{width:640,height:360})
  assert.deepEqual(nativeOutputBlockers(s,{width:640,height:360,fps:30}),[])
  assert.equal(s.vsr_scale,4)
  assert.deepEqual(vsrOutputSize({...s,vsr_mode:'standard'},{width:640,height:360}),{width:2560,height:1440})
})
test('streaming selects only valid quality and defaults preserve old presets',()=>{
  const old=presetSettings({vsr_quality:3})
  assert.equal(old.vsr_mode,'standard');assert.equal(old.vsr_strength,null)
  assert.deepEqual(vsrModePatch('streaming',old),{vsr_mode:'streaming',vsr_quality:2})
  assert.deepEqual(vsrModePatch('streaming',{...old,vsr_quality:4}),{vsr_mode:'streaming'})
  assert.ok(vsrSameSize({vsr_mode:'deblur'}))
  assert.equal(presetValue('vsr_mode','high_bitrate'),'高码率超分辨率')
  assert.equal(presetValue('vsr_strength',.6),'60%')
})
