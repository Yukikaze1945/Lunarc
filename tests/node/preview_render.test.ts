import {test} from 'node:test'
import assert from 'node:assert/strict'
import {previewFrameRange} from '../../frontend/src/services/previewRender.ts'

test('short preview starts at the playhead and clamps to the last source frame',()=>{
  assert.deepEqual(previewFrameRange({frames:240,fps:24},10,'f5'),{start_frame:10,end_frame:15})
  assert.deepEqual(previewFrameRange({frames:240,fps:24},237,'f10'),{start_frame:237,end_frame:240})
  assert.deepEqual(previewFrameRange({frames:240,fps:24},999,'f5'),{start_frame:239,end_frame:240})
})
test('duration is calculated in source frames, with exclusive end and fractional fps',()=>{
  assert.deepEqual(previewFrameRange({frames:1000,fps:60},7,'s0.5'),{start_frame:7,end_frame:37})
  assert.deepEqual(previewFrameRange({frames:1000,fps:29.97},0,'s2'),{start_frame:0,end_frame:60})
  assert.deepEqual(previewFrameRange({frames:1000,fps:24},0,'s5'),{start_frame:0,end_frame:120})
})
