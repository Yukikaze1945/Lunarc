import {test} from 'node:test'
import assert from 'node:assert/strict'
import {PARAMETER_HELP, FLOW_QUALITY_LABELS} from '../../frontend/src/services/parameterHelp.ts'
import {hdrSettingsBlockers} from '../../frontend/src/services/hdrExport.ts'

test('model automatic mask and manual ROI explanations distinguish their scope', () => {
  assert.match(PARAMETER_HELP.use_auto_mask, /模型.*自动遮罩/)
  assert.match(PARAMETER_HELP.roi_kind, /手动.*独立/)
  assert.match(PARAMETER_HELP.temporal, /不包含光流重投影/)
  assert.match(PARAMETER_HELP.total_gain, /不等同于模型 NR 强度/)
})

test('flow labels translate presets without replacing provider IDs', () => {
  assert.equal(FLOW_QUALITY_LABELS.Fast, '速度')
  assert.equal(FLOW_QUALITY_LABELS.MEDIUM, '均衡')
  assert.equal(FLOW_QUALITY_LABELS.Quality, '质量')
  assert.match(PARAMETER_HELP.raft_analysis_edge, /不降低主画面处理分辨率/)
  assert.match(PARAMETER_HELP.flow_quality_raft, /RAFT.*512.*720.*1080/)
  assert.doesNotMatch(PARAMETER_HELP.flow_quality_raft, /AMD|8 × 8/)
  assert.match(PARAMETER_HELP.flow_quality_dis, /OpenCV DIS/)
  assert.equal(FLOW_QUALITY_LABELS.Performance, '性能')
  assert.match(PARAMETER_HELP.flow_quality_amd, /性能.*半分辨率.*质量.*全分辨率/)
  assert.match(PARAMETER_HELP.flow_quality_amd, /不降低主画面分辨率/)
})

test('HDR does not reject supported CUDA/device flow engines or silently replace them', () => {
  for (const flow_engine of ['nvof','fidelityfx','raft','raft_large']) {
    const reasons = hdrSettingsBlockers({experimental_hdr_scrgb:true,guidance_mode:1,flow_engine} as any, null)
    assert.ok(!reasons.some(r => r.startsWith('flow_engine=')))
  }
  const reasons = hdrSettingsBlockers({experimental_hdr_scrgb:true,guidance_mode:1,flow_engine:'dis'} as any, null)
  assert.ok(reasons.some(r => r.startsWith('flow_engine=dis')))
})
