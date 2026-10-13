import test from 'node:test'
import assert from 'node:assert/strict'
import { moveStage, pipelinePatch, renderStages, vsrOutputSize, framegenOutputFps, nativeOutputBlockers } from '../../frontend/src/services/renderPipeline.ts'
import type { DLSSSettings } from '../../frontend/src/types/index.ts'

test('migrate legacy order and honor explicit order', () => {
  assert.deepEqual(renderStages({neural_mode:'nr_dlaa'} as DLSSSettings), ['nr','dlaa'])
  assert.deepEqual(renderStages({neural_mode:'nr_dlaa',render_pipeline:['dlaa','nr']} as DLSSSettings), ['dlaa','nr'])
})
test('move in both directions without mutating source or losing stages', () => {
  const source = ['dlaa','nr','vsr'] as const
  assert.deepEqual(moveStage([...source],0,2), ['nr','vsr','dlaa'])
  assert.deepEqual(moveStage([...source],2,0), ['vsr','dlaa','nr'])
  assert.deepEqual(source, ['dlaa','nr','vsr'])
})
test('native path selected explicitly and standalone stages disable NR scaling', () => {
  assert.equal(pipelinePatch(['vsr','nr']).export_backend, 'gpu_transfer')
  assert.equal(pipelinePatch(['dlaa','vsr']).working_scale, 1)
  assert.equal(pipelinePatch(['vsr']).guidance_mode, 0)
  assert.equal(pipelinePatch(['dlaa','nr']).working_scale, undefined)
})

test('VSR output extents and dynamic fps match queue display and reject invalid output', () => {
  const settings = {render_pipeline:['nr','vsr'],vsr_output_mode:'scale',vsr_scale:1.5,frame_generation:'vfg_dynamic',framegen_target_fps:'60000/1001'} as DLSSSettings
  const video = {width:1280,height:720,fps:25}
  assert.deepEqual(vsrOutputSize(settings,video),{width:1920,height:1080})
  assert.equal(framegenOutputFps(settings,25),60000/1001)
  assert.deepEqual(nativeOutputBlockers(settings,video),[])
  assert.ok(nativeOutputBlockers({...settings,framegen_target_fps:24},video).length)
  assert.ok(nativeOutputBlockers({...settings,vsr_scale:5},video).length)
  assert.deepEqual(vsrOutputSize({...settings,vsr_output_mode:'resolution',vsr_target:'custom',vsr_width:2560,vsr_height:1440},video),{width:2560,height:1440})
  assert.equal(framegenOutputFps({frame_generation:'vfg',frame_multiplier:8} as DLSSSettings,25),200)
})

test('VFG is movable and independently usable; adding and removing controls activation', () => {
  const old = {render_pipeline:['nr','vsr'],frame_generation:'vfg_dynamic'} as DLSSSettings
  assert.deepEqual(renderStages(old), ['nr','vsr','vfg'])
  assert.deepEqual(moveStage(renderStages(old),2,0), ['vfg','nr','vsr'])
  assert.equal(pipelinePatch(['vfg'],old).frame_generation,'vfg_dynamic')
  assert.equal(pipelinePatch(['vfg'],old).guidance_mode,0)
  assert.equal(pipelinePatch(['nr'],old).frame_generation,'off')
  assert.equal(pipelinePatch(['vfg']).frame_generation,'vfg')
})

test('TrueHDR is one named sortable stage; legacy flags and explicit order agree', () => {
  const legacy = {render_pipeline:['dlaa','nr'],truehdr_enabled:true} as DLSSSettings
  assert.deepEqual(renderStages(legacy), ['dlaa','nr','truehdr'])
  const moved = moveStage(renderStages(legacy),2,0)
  assert.deepEqual(moved, ['truehdr','dlaa','nr'])
  const patch = pipelinePatch(moved,legacy)
  assert.equal(patch.truehdr_enabled,true)
  assert.deepEqual(renderStages({...legacy,...patch}), moved)
  assert.equal(pipelinePatch(['nr'],legacy).truehdr_enabled,false)
  assert.deepEqual(renderStages({render_pipeline:['nr','truehdr'],truehdr_enabled:true} as DLSSSettings), ['nr','truehdr'])
})
