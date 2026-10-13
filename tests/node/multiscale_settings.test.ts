import test from 'node:test'
import assert from 'node:assert/strict'
import { DEFAULT_SETTINGS, completeSettings } from '../../frontend/src/services/workspace.ts'
import { chooseReconstructionMode, chooseFusionMode, changeLayerNR, layerNRSettings } from '../../frontend/src/services/multiScaleSettings.ts'

test('single to dual/triple selects sequential sessions at the requested scale', () => {
  for (const mode of ['dual', 'triple'] as const) {
    const next = {...DEFAULT_SETTINGS, ...chooseReconstructionMode(DEFAULT_SETTINGS, mode)}
    assert.equal(next.fusion_mode, 'sequential')
    assert.deepEqual(next.layer_scales,mode === 'dual' ? [1,1] : [1,1,1])
    const multi = chooseFusionMode(next,'data_consistency_experimental')
    assert.equal(new Set(multi.layer_scales).size, mode === 'dual' ? 2 : 3)
  }
})

test('independent sliders preserve other layers even when L1 subsequently changes', () => {
  const triple = {...DEFAULT_SETTINGS, ...chooseReconstructionMode(DEFAULT_SETTINGS, 'triple')}
  const second = {...triple, ...changeLayerNR(triple, 1, 'intensity', .4)}
  const third = {...second, ...changeLayerNR(second, 2, 'local_tone', .2)}
  const first = {...third, ...changeLayerNR(third, 0, 'intensity', 1.8)}
  assert.equal(layerNRSettings(first, 0).intensity, 1.8)
  assert.equal(layerNRSettings(first, 1).intensity, .4)
  assert.equal(layerNRSettings(first, 2).intensity, triple.intensity)
  assert.equal(layerNRSettings(first, 2).local_tone, .2)
  assert.equal(layerNRSettings(triple, 1).intensity, triple.intensity)
})

test('legacy workspace migrates once; explicit new comparison stays selected', () => {
  const legacy = {...DEFAULT_SETTINGS, reconstruction_mode:'dual' as const, layer_scales:[1, .75]}
  const updated = completeSettings(legacy)
  assert.equal(updated.fusion_mode, 'sequential')
  assert.equal(completeSettings({...updated, fusion_mode:'catmull_baseline'}).fusion_mode, 'catmull_baseline')
})
