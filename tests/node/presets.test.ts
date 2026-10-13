import {jobComparison} from '../../frontend/src/services/jobComparison.ts'
import {test} from 'node:test'
import assert from 'node:assert/strict'
import {DEFAULT_PRESET,presetChanges,presetSettings,samePreset} from '../../frontend/src/services/presets.ts'

test('only actual changes are listed, with the numeric amount against defaults',()=>{
  assert.deepEqual(presetChanges(DEFAULT_PRESET),[])
  const changes=presetChanges({...DEFAULT_PRESET,intensity:.7,style:2})
  assert.equal(changes.length,2)
  const strength=changes.find(c=>c.key==='intensity')!
  assert.ok(Math.abs(strength.delta!+.3)<1e-10)
  assert.ok(Math.abs(strength.percent!+30)<1e-10)
  assert.equal(changes.find(c=>c.key==='style')!.after,'电影')
  assert.equal(changes.find(c=>c.key==='style')!.delta,undefined)
})
test('applying the default snapshot clears legacy extra stages and layer overrides',()=>{
  assert.deepEqual(DEFAULT_PRESET.render_pipeline,['nr'])
  assert.equal(DEFAULT_PRESET.truehdr_enabled,false)
  assert.deepEqual(DEFAULT_PRESET.layer_settings,[])
  assert.deepEqual(presetSettings({truehdr_enabled:true}).render_pipeline,['nr','truehdr'])
})
test('layer numeric changes stay separate and object key order does not matter',()=>{
  const snapshot={...DEFAULT_PRESET,layer_settings:[{}, {intensity:.8}],layer_scales:[1,.5]}
  assert.match(presetChanges(snapshot).find(c=>c.key==='第 2 层.intensity')!.label,/第 2 层/)
  assert.equal(presetChanges(snapshot).find(c=>c.key==='第 2 层.layer_scales')!.percent,-50)
  assert.ok(samePreset(DEFAULT_PRESET,Object.fromEntries(Object.entries(DEFAULT_PRESET).reverse())))
})

test('comparison links from the queue use the selected source interval',()=>{
  assert.deepEqual(jobComparison({input_path:'source',output_path:'clip',frame_range:{start_frame:90,end_frame:150},source_fps:30}),
    {original:'source',exported:'clip',original_offset_seconds:3,duration_seconds:2})
})
