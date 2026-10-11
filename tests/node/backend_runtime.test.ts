import {test} from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import {runtimeCandidates, runtimeEnvironment, runtimeFingerprint, probeRuntime} from '../../frontend/electron/backendRuntime.ts'

test('published package ignores developer Python and isolates system dependencies', () => {
  const root = path.resolve('fake-installed-resources')
  assert.deepEqual(runtimeCandidates(root, true, {PYTHON_PATH: 'foreign-python'}), [path.join(root, 'runtime/python/python.exe')])
  const env = runtimeEnvironment(root, 'user-data', {PYTHONPATH: 'foreign-libs', PYTHONHOME: 'foreign-root', FFMPEG_SDK_DIR: 'foreign-ffmpeg', PATH: 'foreign-path'}, true)
  assert.equal(env.PYTHONPATH, undefined)
  assert.equal(env.PYTHONHOME, undefined)
  assert.equal(env.FFMPEG_SDK_DIR, undefined)
  assert.ok(!env.PATH?.includes('foreign-path'))
  assert.equal(env.LUNARC_STANDALONE, '1')
})

test('development override is explicit and missing executables fail the actual probe', async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'lunarc-runtime-'))
  try {
    const missing = path.join(root, 'missing-python.exe')
    assert.deepEqual(runtimeCandidates(root, false, {PYTHON_PATH: missing}), [missing])
    await assert.rejects(probeRuntime(missing, root, {}), /missing-python/)
  } finally {fs.rmSync(root, {recursive: true})}
})

test('backend reuse distinguishes changed source even with an unchanged runtime', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'lunarc-fingerprint-'))
  try {
    const runtime = path.join(root, 'runtime')
    fs.mkdirSync(runtime)
    fs.writeFileSync(path.join(runtime, 'manifest.json'), '{}')
    fs.writeFileSync(path.join(root, 'server.py'), 'version = 1')
    const previous = runtimeFingerprint(root, runtime)
    fs.writeFileSync(path.join(root, 'server.py'), 'version = 2')
    assert.notEqual(runtimeFingerprint(root, runtime), previous)
    fs.writeFileSync(path.join(root, 'server.py'), 'version = 1')
    assert.equal(runtimeFingerprint(root, runtime), previous)
  } finally {fs.rmSync(root, {recursive: true})}
})
